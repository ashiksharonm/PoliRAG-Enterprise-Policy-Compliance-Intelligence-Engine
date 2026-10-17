"""Policy change impact analysis: parsing, diff, retrieval, analysis, answers, service."""

import importlib.util
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.impact import answer as ans
from src.impact.analysis import (
    ImpactConfig,
    Question,
    analyze_impact,
    corpus_from_texts,
    evaluate_against_gold,
    load_corpus,
    load_questions,
    oracle_affected,
    render_markdown,
)
from src.impact.clauses import parse_policy
from src.impact.diff import diff_docs
from src.impact.embedders import CachedEmbedder, HashingEmbedder, make_embedder
from src.impact.index import ClauseIndex
from src.impact.service import create_app
from src.impact.settings import Settings

ROOT = Path(__file__).resolve().parents[1]

DOC = """ACME Test Policy
Document ID: POL-T-001
Effective Date: May 1, 2025

1. PURPOSE

This policy sets rules for tests.

2. RETENTION

2.1 Invoices
Invoices must be retained for seven (7) years.

2.2 Receipts
Receipts must be retained for one (1) year. Scans are acceptable.

3. ACCESS

3.1 Passwords
Passwords must be at least 12 characters long.
"""


def _doc(text=DOC, name="t.txt"):
    return parse_policy(text, name)


# ------------------------------------------------------------------ clauses
def test_parse_policy_clauses_ids_and_metadata():
    d = _doc()
    assert d.doc_id == "POL-T-001" and d.effective_date == "May 1, 2025"
    assert [c.id for c in d.clauses] == [
        "POL-T-001#1",
        "POL-T-001#2.1",
        "POL-T-001#2.2",
        "POL-T-001#3.1",
    ]
    assert d.clauses[0].title == "Purpose" and "rules for tests" in d.clauses[0].body
    assert d.clauses[1].text.startswith("2.1 Invoices.")


def test_parse_policy_without_document_id_uses_name_and_letter_suffix():
    d = parse_policy("Title\n\n1. SCOPE\n\n1.1 A\nText a.\n\n1.1a B\nText b.\n", "my file")
    assert d.doc_id == "my-file" and [c.number for c in d.clauses] == ["1.1", "1.1a"]


def test_content_hash_ignores_number_and_whitespace():
    a = _doc().clauses[1]
    b = parse_policy(
        DOC.replace("2.1 Invoices", "9.9 Invoices").replace("Invoices must", "Invoices   must")
    )
    assert a.content_hash == b.clauses[1].content_hash


# --------------------------------------------------------------------- diff
def test_diff_modified_added_removed_unchanged():
    new_text = (
        DOC.replace("seven (7)", "ten (10)").replace(
            "2.2 Receipts\nReceipts must be retained for one (1) year. Scans are acceptable.\n\n",
            "",
        )
        + "\n3.2 MFA\nMulti-factor authentication is required for all remote logins.\n"
    )
    ch = {(c.kind, (c.new or c.old).number): c for c in diff_docs(_doc(), _doc(new_text))}
    assert (
        ch[("modified", "2.1")].word_changes == [("seven (7) years.", "ten (10) years.")]
        or ch[("modified", "2.1")].word_changes
    )
    assert ("removed", "2.2") in ch and ("added", "3.2") in ch
    assert ("unchanged", "3.1") in ch and ("unchanged", "1") in ch
    assert ch[("modified", "2.1")].to_dict()["word_changes"]


def test_diff_detects_renumbering_and_retitled_edit():
    renum = DOC.replace("2.1 Invoices", "2.0 Invoices")
    kinds = {c.kind for c in diff_docs(_doc(), _doc(renum))}
    assert "renumbered" in kinds and "modified" not in kinds
    retitled = DOC.replace("2.1 Invoices", "2.1 Invoice retention").replace(
        "seven (7)", "eight (8)"
    )
    changes = diff_docs(_doc(), _doc(retitled))
    mod = [c for c in changes if c.kind == "modified"]
    assert len(mod) == 1 and mod[0].old.number == "2.1" and mod[0].new.title == "Invoice retention"


def test_diff_identical_has_no_changes():
    assert all(c.kind == "unchanged" for c in diff_docs(_doc(), _doc()))


# ----------------------------------------------------------- embed / index
def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(64)
    a, b = e.encode(["data retention period"]), e.encode(["data retention period"])
    assert np.allclose(a, b) and abs(float(np.linalg.norm(a[0])) - 1) < 1e-5
    assert np.all(e.encode([""]) == 0)


def test_cached_embedder_encodes_each_text_once():
    calls = []

    class Spy:
        name = "spy"

        def encode(self, texts):
            calls.append(list(texts))
            return np.ones((len(texts), 2), dtype=np.float32)

    c = CachedEmbedder(Spy(), max_items=2)
    c.encode(["a", "b", "a"])
    c.encode(["a"])
    assert calls == [["a", "b"]]
    c.encode(["c", "d", "e"])  # exceeds max_items: cache is reset, still correct
    assert c.encode([]).shape[0] == 0


def test_make_embedder_rejects_unknown_backend():
    with pytest.raises(ValueError):
        make_embedder("nope", "x")
    assert make_embedder("hashing", "").name.startswith("hashing")


def test_index_search_and_reranker_order():
    docs = [_doc()]
    idx = ClauseIndex(docs, HashingEmbedder())
    top = idx.search("how many characters must a password have", k=2)
    assert top[0].clause.id == "POL-T-001#3.1" and top[0].rank == 1

    class Reverse:
        def scores(self, q, passages):
            return [-float(i) for i in range(len(passages))]

    first = idx.search("password", k=1)[0].clause.id
    rr = ClauseIndex(docs, HashingEmbedder(), Reverse(), candidates=4)
    assert rr.search("password", k=1)[0].clause.id == first
    assert ClauseIndex([], HashingEmbedder()).search("x") == []


# ----------------------------------------------------------------- analysis
def _pair(new_text):
    return corpus_from_texts({"t.txt": DOC}), corpus_from_texts({"t.txt": new_text})


QS = [
    Question(
        "a",
        "How many years must invoices be retained?",
        "numeric",
        [("t.txt", "Invoices must be retained for seven (7) years")],
    ),
    Question(
        "b", "How long must passwords be?", "factual", [("t.txt", "at least 12 characters long")]
    ),
    Question("c", "What is the dress code?", "not_covered"),
]


def test_impact_flags_only_questions_touching_changed_clauses():
    old, new = _pair(DOC.replace("seven (7)", "ten (10)"))
    r = analyze_impact(old, new, QS, HashingEmbedder())
    flagged = {x["id"] for x in r["questions"] if x["likely_affected"]}
    assert "a" in flagged and "b" not in flagged
    a = next(x for x in r["questions"] if x["id"] == "a")
    assert a["severity"] == "high" and a["reasons"][0]["reason"] == "evidence_modified"
    assert a["reasons"][0]["old_id"] == "POL-T-001#2.1"
    assert "[POL-T-001#2.1]" in a["answer_old"]["text"] and "ten (10)" in a["answer_new"]["text"]
    assert r["summary"]["clauses"]["modified"] == 1
    gold = evaluate_against_gold(r, old, new, QS)
    assert gold["tp"] == 1 and gold["fn"] == 0 and gold["recall"] == 1.0
    assert "Answers likely affected" in render_markdown(r)


def test_identical_versions_affect_nothing():
    old, new = _pair(DOC)
    r = analyze_impact(old, new, QS, HashingEmbedder())
    assert r["summary"]["n_likely_affected"] == 0
    assert "None." in render_markdown(r)


def test_removed_and_added_clauses():
    removed = DOC.replace("3.1 Passwords\nPasswords must be at least 12 characters long.\n", "")
    old, new = _pair(removed)
    r = analyze_impact(old, new, QS, HashingEmbedder(), config=ImpactConfig(refine=False))
    b = next(x for x in r["questions"] if x["id"] == "b")
    assert any(x["reason"] == "evidence_removed" for x in b["reasons"])
    assert oracle_affected(old, new, QS[1]) is True and oracle_affected(old, new, QS[2]) is None
    added = DOC + "\n3.2 Lockout\nPasswords lock after five failed attempts.\n"
    lock = [Question("l", "After how many failed attempts do passwords lock?")]
    r2 = analyze_impact(
        *_pair(added),
        lock,
        HashingEmbedder(),
        config=ImpactConfig(refine=False, min_severity="low"),
    )
    assert any(x["reason"] == "evidence_added" for q in r2["questions"] for x in q["reasons"])


def test_wording_only_edit_is_filtered_by_refinement():
    old, new = _pair(DOC.replace("Scans are acceptable.", "Scans are fine."))
    q = [Question("r", "Receipts must be retained for how many years?")]
    assert (
        analyze_impact(old, new, q, HashingEmbedder(), config=ImpactConfig(refine=False))[
            "summary"
        ]["n_likely_affected"]
        == 1
    )
    assert (
        analyze_impact(old, new, q, HashingEmbedder(), config=ImpactConfig(refine=True, k=1))[
            "summary"
        ]["n_likely_affected"]
        == 0
    )


def test_invalid_min_severity_and_empty_questions():
    old, new = _pair(DOC)
    with pytest.raises(ValueError):
        analyze_impact(old, new, [], config=ImpactConfig(min_severity="x"))
    assert analyze_impact(old, new, [])["summary"]["n_questions"] == 0


def test_bundled_v1_v2_end_to_end_counts_and_known_hit():
    old, new = load_corpus(ROOT / "data/raw"), load_corpus(ROOT / "data/versions/v2")
    qs = load_questions(ROOT / "data/eval/retrieval_gold.json")
    assert len(qs) == 76
    r = analyze_impact(old, new, qs, HashingEmbedder())
    assert r["summary"]["clauses"] == {
        "unchanged": 84,
        "renumbered": 0,
        "modified": 5,
        "added": 1,
        "removed": 1,
    }
    q023 = next(x for x in r["questions"] if x["id"] == "q023")  # financial records retention
    assert q023["likely_affected"] and "POL-2025-001#3.1" in json.dumps(q023["reasons"])


def test_v2_corpus_is_reproducible_from_script(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("mk", ROOT / "scripts/make_v2_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "DST", tmp_path)
    mod.main()
    for f in (ROOT / "data/versions/v2").glob("*.txt"):
        assert (tmp_path / f.name).read_text() == f.read_text()


# ------------------------------------------------------------------- answers
HITS = ClauseIndex([_doc()], HashingEmbedder()).search("how many years invoices retained", k=2)


def test_extractive_answer_cites_and_handles_no_hits():
    a = ans.extractive_answer("How many years must invoices be retained?", HITS)
    assert a.mode == "extractive" and a.citations[0] == "POL-T-001#2.1"
    assert "[POL-T-001#2.1]" in a.text and "seven (7) years" in a.text
    assert ans.extractive_answer("x", []).citations == []


def test_validate_citations():
    assert ans.validate_citations("Seven years [POL-T-001#2.1].", HITS) == ["POL-T-001#2.1"]
    assert ans.validate_citations("Seven years.", HITS) is None
    assert ans.validate_citations("Seven [OTHER-1#9.9].", HITS) is None
    assert ans.validate_citations("Not covered by the provided clauses.", HITS) == []


class _Gen(ans.LLMGenerator):
    def __init__(self, reply=None, exc=None):
        super().__init__("groq", "k", "m")
        self.reply, self.exc = reply, exc

    def complete(self, prompt):
        if self.exc:
            raise self.exc
        return self.reply


def test_llm_answer_used_only_with_valid_citations():
    ok = ans.answer_question("q", HITS, _Gen("Seven years [POL-T-001#2.1]."))
    assert ok.mode == "groq" and ok.citations == ["POL-T-001#2.1"]
    for bad in (
        _Gen("No citation here."),
        _Gen(exc=httpx.ConnectError("down")),
        _Gen("Bad [POL-T-001#9.9]."),
    ):
        r = ans.answer_question("How many years must invoices be retained?", HITS, bad)
        assert r.mode == "extractive-fallback" and r.citations


def test_prompt_lists_clause_ids():
    assert "[POL-T-001#2.1]" in ans.build_prompt("q?", HITS)


def test_generator_from_env_and_http_calls(monkeypatch):
    for k in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert ans.generator_from_env() is None
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    gem = ans.generator_from_env()
    assert gem.provider == "gemini"
    monkeypatch.setenv("GROQ_API_KEY", "x")
    groq = ans.generator_from_env()
    assert groq.provider == "groq"

    def fake_post(url, **kw):
        body = (
            {"choices": [{"message": {"content": " hi [A#1] "}}]}
            if "groq" in url
            else {"candidates": [{"content": {"parts": [{"text": " yo "}]}}]}
        )
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(ans.httpx, "post", fake_post)
    assert groq.complete("p") == "hi [A#1]" and gem.complete("p") == "yo"
    with pytest.raises(ValueError):
        ans.LLMGenerator("other", "k", "m")


# ------------------------------------------------------------------- service
@pytest.fixture()
def client():
    return TestClient(create_app(Settings(embedding_backend="hashing", log_level="WARNING")))


def test_health_ready_versions_and_request_id(client):
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/readyz").json()["status"] == "ready"
    r = client.get("/v1/versions", headers={"x-request-id": "abc"})
    assert r.headers["x-request-id"] == "abc" and r.json()["v1"]["documents"] == 7
    assert "x-request-id" in client.get("/healthz").headers


def test_diff_and_impact_endpoints(client):
    d = client.post("/v1/diff", json={"old": {"version": "v1"}, "new": {"version": "v2"}}).json()
    assert d["counts"]["modified"] == 5 and d["counts"]["removed"] == 1
    r = client.post("/v1/impact", json={"old": {"version": "v1"}, "new": {"version": "v2"}})
    assert r.status_code == 200 and r.json()["summary"]["n_questions"] == 76
    custom = client.post(
        "/v1/impact",
        json={
            "old": {"files": {"p.txt": DOC}},
            "new": {"files": {"p.txt": DOC.replace("seven", "ten")}},
            "questions": [{"text": "How many years must invoices be retained?"}],
        },
    ).json()
    assert custom["summary"]["n_likely_affected"] == 1


def test_ask_endpoint_cites_clauses(client):
    r = client.post(
        "/v1/ask",
        json={
            "question": "What is the data subject request deadline?",
            "corpus": {"version": "v2"},
        },
    ).json()
    assert r["answer"]["mode"] == "extractive" and r["answer"]["citations"]
    assert r["evidence"][0]["clause_id"].startswith("POL-2025-")


def test_validation_errors_and_limits(client):
    assert client.post("/v1/diff", json={"old": {}, "new": {"version": "v1"}}).status_code == 422
    assert (
        client.post(
            "/v1/diff", json={"old": {"version": "v9"}, "new": {"version": "v1"}}
        ).status_code
        == 404
    )
    assert client.post("/v1/ask", json={"question": "x"}).status_code == 422
    small = TestClient(
        create_app(
            Settings(
                embedding_backend="hashing",
                max_request_bytes=10,
                max_questions=1,
                log_level="WARNING",
            )
        )
    )
    assert (
        small.post(
            "/v1/diff", json={"old": {"files": {"a": "x" * 50}}, "new": {"version": "v1"}}
        ).status_code
        == 413
    )
    two = [{"text": "question one"}, {"text": "question two"}]
    assert (
        small.post(
            "/v1/impact",
            json={"old": {"version": "v1"}, "new": {"version": "v2"}, "questions": two},
        ).status_code
        == 413
    )


def test_api_key_enforced_but_health_open():
    c = TestClient(
        create_app(Settings(embedding_backend="hashing", api_key="s3cret", log_level="WARNING"))
    )
    assert c.get("/healthz").status_code == 200
    assert c.get("/v1/versions").status_code == 401
    assert c.get("/v1/versions", headers={"x-api-key": "s3cret"}).status_code == 200


def test_readyz_fails_when_corpora_missing(tmp_path):
    c = TestClient(
        create_app(
            Settings(embedding_backend="hashing", v1_dir=tmp_path / "no", log_level="WARNING")
        )
    )
    assert c.get("/readyz").status_code == 503


def test_cli_impact_and_diff_and_ask(capsys, tmp_path):
    from src.impact.cli import main

    assert main(["diff"]) == 0
    assert "modified" in capsys.readouterr().out
    out = tmp_path / "r.json"
    assert main(["impact", "--backend", "hashing", "--score-vs-gold", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "Policy change impact report" in text and "Gold-oracle evaluation" in text
    assert json.loads(out.read_text())["summary"]["n_questions"] == 76
    assert main(["ask", "How long are backups retained?", "--backend", "hashing"]) == 0
    assert "citations" in capsys.readouterr().out

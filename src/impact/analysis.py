"""Policy change impact analysis.

Given two versions of a corpus and a set of questions, retrieve evidence for every question
in both versions and report which answers are likely affected, citing clause ids.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .answer import Answer, LLMGenerator, answer_question, extractive_answer
from .clauses import Clause, PolicyDoc, parse_policy
from .diff import ClauseChange, diff_corpus
from .embedders import Embedder, HashingEmbedder
from .index import ClauseIndex, Hit, Reranker

SEVERITY_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}


@dataclass
class Question:
    id: str
    text: str
    type: str = ""
    gold_quotes: list[tuple[str, str]] = field(default_factory=list)  # (doc file name, quote)


@dataclass
class Corpus:
    docs: list[PolicyDoc]
    files: dict[str, PolicyDoc]  # file name -> doc
    raw: dict[str, str]  # file name -> raw text

    def text_for(self, doc_id: str) -> str:
        return next((self.raw[n] for n, d in self.files.items() if d.doc_id == doc_id), "")


def load_corpus(path: str | Path) -> Corpus:
    raw = {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(path).glob("*.txt"))}
    return corpus_from_texts(raw)


def corpus_from_texts(raw: dict[str, str]) -> Corpus:
    files = {name: parse_policy(text, Path(name).stem) for name, text in raw.items()}
    return Corpus(list(files.values()), files, raw)


def load_questions(gold_path: str | Path) -> list[Question]:
    """Read ``data/eval/retrieval_gold.json``; questions with no evidence keep an empty quote list."""
    data = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    return [
        Question(
            q["id"],
            q["question"],
            q.get("type", ""),
            [(e["doc"], e["quote"]) for e in q.get("evidence", [])],
        )
        for q in data["questions"]
    ]


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def quote_present(raw_text: str, quote: str) -> bool:
    return _squash(quote) in _squash(raw_text)


def _clause_json(c: Clause) -> dict[str, str]:
    return {"id": c.id, "title": c.title}


def _hit_json(h: Hit) -> dict[str, Any]:
    return {
        "clause_id": h.clause.id,
        "title": h.clause.title,
        "rank": h.rank,
        "score": round(h.score, 4),
    }


def _strip_cites(text: str) -> str:
    return _squash(re.sub(r"\[[^\]]*#[0-9.a-z]+\]", "", text))


class _Findings:
    """Reasons collected for one question and the highest severity among them."""

    def __init__(self) -> None:
        self.reasons: list[dict[str, Any]] = []
        self.severity = "none"
        self._seen: set[tuple[str, str | None, str | None]] = set()

    def bump(self, level: str) -> None:
        if SEVERITY_ORDER[level] > SEVERITY_ORDER[self.severity]:
            self.severity = level

    def add(self, kind: str, ch: ClauseChange, hit: Hit, version: str) -> None:
        key = (kind, ch.old.id if ch.old else None, ch.new.id if ch.new else None)
        if key in self._seen:
            return
        self._seen.add(key)
        self.reasons.append({"reason": kind, "version": version, "rank": hit.rank, **ch.to_dict()})
        self.bump("high" if hit.rank == 1 else "medium")


@dataclass
class ImpactConfig:
    k: int = 2  # evidence depth: the top-k clauses are treated as the answer's evidence
    min_severity: str = "medium"  # lowest severity counted as "likely affected"
    refine: bool = True  # drop findings where the extractive answer text is identical
    with_answers: bool = True


def analyze_impact(
    old: Corpus,
    new: Corpus,
    questions: Sequence[Question],
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    config: ImpactConfig | None = None,
    generator: LLMGenerator | None = None,
) -> dict[str, Any]:
    cfg = config or ImpactConfig()
    if cfg.min_severity not in ("low", "medium", "high"):
        raise ValueError("min_severity must be low, medium or high")
    emb = embedder or HashingEmbedder()
    changes = diff_corpus(old.docs, new.docs)
    by_old = {c.old.id: c for c in changes if c.old}
    by_new = {c.new.id: c for c in changes if c.new}
    idx_old, idx_new = ClauseIndex(old.docs, emb, reranker), ClauseIndex(new.docs, emb, reranker)
    qvecs = emb.encode([q.text for q in questions]) if questions else np.zeros((0, 1))
    threshold = SEVERITY_ORDER[cfg.min_severity]

    rows: list[dict[str, Any]] = []
    for q, qv in zip(questions, qvecs, strict=True):
        ho, hn = idx_old.search(q.text, cfg.k, qv), idx_new.search(q.text, cfg.k, qv)
        found = _Findings()

        for h in ho:
            ch = by_old[h.clause.id]
            if ch.kind in ("modified", "removed"):
                found.add(f"evidence_{ch.kind}", ch, h, "old")
        for h in hn:
            ch = by_new[h.clause.id]
            if ch.kind in ("modified", "added"):
                found.add(f"evidence_{ch.kind}", ch, h, "new")
        if not found.reasons:
            canon_old = [n.id if (n := by_old[h.clause.id].new) is not None else None for h in ho]
            if canon_old != [h.clause.id for h in hn]:
                found.reasons.append(
                    {
                        "reason": "evidence_reordered",
                        "old_ids": [h.clause.id for h in ho],
                        "new_ids": [h.clause.id for h in hn],
                    }
                )
                found.bump("low")

        reasons, severity = found.reasons, found.severity
        row: dict[str, Any] = {
            "id": q.id,
            "question": q.text,
            "type": q.type,
            "severity": severity,
            "evidence_old": [_hit_json(h) for h in ho],
            "evidence_new": [_hit_json(h) for h in hn],
            "reasons": reasons,
        }
        ext_old, ext_new = extractive_answer(q.text, ho), extractive_answer(q.text, hn)
        answer_changed = _strip_cites(ext_old.text) != _strip_cites(ext_new.text)
        row["answer_changed"] = answer_changed
        affected = SEVERITY_ORDER[severity] >= threshold and (answer_changed or not cfg.refine)
        row["likely_affected"] = affected
        if affected and cfg.with_answers:
            a_old: Answer = answer_question(q.text, ho, generator)
            a_new: Answer = answer_question(q.text, hn, generator)
            row["answer_old"], row["answer_new"] = a_old.to_dict(), a_new.to_dict()
        rows.append(row)

    affected_rows = [r for r in rows if r["likely_affected"]]
    return {
        "config": {
            "k": cfg.k,
            "min_severity": cfg.min_severity,
            "refine": cfg.refine,
            "embedder": emb.name,
            "reranker": reranker is not None,
        },
        "summary": {
            "n_questions": len(rows),
            "n_likely_affected": len(affected_rows),
            "by_severity": {
                s: sum(r["severity"] == s for r in rows) for s in ("high", "medium", "low", "none")
            },
            "clauses": {
                k: sum(c.kind == k for c in changes)
                for k in ("unchanged", "renumbered", "modified", "added", "removed")
            },
        },
        "clause_changes": [c.to_dict() for c in changes if c.kind != "unchanged"],
        "questions": rows,
    }


def oracle_affected(old: Corpus, new: Corpus, q: Question) -> bool | None:
    """Gold-based ground truth: is a gold evidence phrase gone from the new version?

    None for questions without gold evidence. A phrase counts as changed when its (document, quote)
    no longer appears verbatim in the new version of that document.
    """
    if not q.gold_quotes:
        return None
    for fname, quote in q.gold_quotes:
        if fname not in new.raw or not quote_present(new.raw[fname], quote):
            return True
    return False


def evaluate_against_gold(
    report: dict[str, Any], old: Corpus, new: Corpus, questions: Sequence[Question]
) -> dict[str, Any]:
    """Precision/recall of ``likely_affected`` vs. the gold-evidence oracle (answerable questions)."""
    pred = {r["id"]: r["likely_affected"] for r in report["questions"]}
    tp = fp = fn = tn = 0
    for q in questions:
        truth = oracle_affected(old, new, q)
        if truth is None:
            continue
        p = pred[q.id]
        tp += p and truth
        fp += p and not truth
        fn += (not p) and truth
        tn += (not p) and not truth
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    return {
        "n_answerable": tp + fp + fn + tn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": None if prec is None else round(prec, 3),
        "recall": None if rec is None else round(rec, 3),
    }


def render_markdown(report: dict[str, Any], max_rows: int = 25) -> str:
    s, c = report["summary"], report["summary"]["clauses"]
    out = [
        "# Policy change impact report",
        "",
        f"- Questions analysed: {s['n_questions']}; **likely affected: {s['n_likely_affected']}**",
        f"- Severity: high {s['by_severity']['high']}, medium {s['by_severity']['medium']}, "
        f"low {s['by_severity']['low']}, none {s['by_severity']['none']}",
        f"- Clauses: {c['modified']} modified, {c['added']} added, {c['removed']} removed, "
        f"{c['renumbered']} renumbered, {c['unchanged']} unchanged",
        f"- Retrieval: {report['config']['embedder']}, evidence depth k={report['config']['k']}, "
        f"reranker={'on' if report['config']['reranker'] else 'off'}",
        "",
        "## Clause changes",
        "",
        "| change | clause | similarity | what changed |",
        "|---|---|---|---|",
    ]
    for ch in report["clause_changes"]:
        cid = ch["new_id"] or ch["old_id"]
        wc = "; ".join(
            f"`{w['old'] or '(none)'}` -> `{w['new'] or '(none)'}`" for w in ch["word_changes"][:3]
        )
        out.append(f"| {ch['kind']} | {cid} ({ch['title']}) | {ch['similarity']} | {wc} |")
    out += ["", "## Answers likely affected", ""]
    rows = [r for r in report["questions"] if r["likely_affected"]]
    if not rows:
        out.append("None.")
    for r in rows[:max_rows]:
        cites = ", ".join(
            sorted({x for rs in r["reasons"] for x in (rs.get("old_id"), rs.get("new_id")) if x})
        )
        out.append(
            f"- **{r['id']}** ({r['severity']}): {r['question']} Evidence changed in: {cites}"
        )
        if "answer_old" in r:
            out.append(f"  - before: {r['answer_old']['text']}")
            out.append(f"  - after: {r['answer_new']['text']}")
    if len(rows) > max_rows:
        out.append(f"- ... and {len(rows) - max_rows} more")
    return "\n".join(out) + "\n"

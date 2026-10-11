#!/usr/bin/env python
"""Retrieval ablation on the policy gold set: BM25 / dense / hybrid / hybrid + rerank.

Runs fully locally (no API keys) with sentence-transformers by default. Retrieval
only: no generation or faithfulness is evaluated.

    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --embedding-backend openai --embedding-model text-embedding-3-small
"""

import argparse
import json
import platform
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.retrieval_metrics import (  # noqa: E402
    bootstrap_ci, fuse, ndcg_at_k, overlaps, recall_at_k, reciprocal_rank,
    resolve_quote,
)

METHODS = ["bm25", "dense", "hybrid", "hybrid+rerank"]
KS = (1, 5, 10)
CANDIDATES = 20  # mirrors RETRIEVAL_TOP_K; the reranker sees the hybrid top-20
SEM_W, BM25_W = 0.7, 0.3  # mirrors RETRIEVAL_SEMANTIC_WEIGHT / RETRIEVAL_BM25_WEIGHT


# ---------------------------------------------------------------- chunking
def chunk_clauses(doc, text):
    """One chunk per numbered clause (3.1, 4.2 ...); bare section headings merge forward."""
    starts = [m.start() for m in re.finditer(r"^(?:\d+\.\d+ |\d+\. [A-Z])", text, re.M)]
    bounds = sorted(set([0] + starts + [len(text)]))
    spans = list(zip(bounds[:-1], bounds[1:]))
    merged, carry = [], None
    for s, e in spans:
        if carry is not None:
            s, carry = carry, None
        if re.fullmatch(r"\d+\. [A-Z ,&-]+\s*", text[s:e]):
            carry = s
            continue
        merged.append((s, e))
    return [(doc, s, e, text[s:e].strip()) for s, e in merged if text[s:e].strip()]


def chunk_windows(doc, text, size=100, stride=75):
    """Fixed word windows with overlap, ignoring document structure."""
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    out = []
    for i in range(0, len(words), stride):
        w = words[i:i + size]
        s, e = w[0][0], w[-1][1]
        out.append((doc, s, e, text[s:e]))
        if i + size >= len(words):
            break
    return out


CHUNKERS = {"clause": chunk_clauses, "window": chunk_windows}


# ---------------------------------------------------------------- embeddings
def make_embedder(backend, model_name):
    if backend == "sentence-transformers":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return lambda texts: np.asarray(
            model.encode(texts, normalize_embeddings=True, batch_size=32,
                         show_progress_bar=False), dtype=np.float32)
    if backend == "openai":
        from openai import OpenAI  # needs OPENAI_API_KEY
        client = OpenAI()

        def embed(texts):
            resp = client.embeddings.create(model=model_name, input=list(texts))
            v = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
            return v / np.linalg.norm(v, axis=1, keepdims=True)
        return embed
    raise ValueError(f"unknown backend {backend}")


# ---------------------------------------------------------------- evaluation
def bm25_scores(bm25, query):
    # repo BM25Store tokenisation: lowercase whitespace split
    return bm25.get_scores(query.lower().split())


def rank_all(chunks, embed, reranker, questions):
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([c[3].lower().split() for c in chunks])
    cvec = embed([c[3] for c in chunks])
    qvec = embed([q["question"] for q in questions])
    rankings = {m: [] for m in METHODS}
    for qi, q in enumerate(questions):
        b = bm25_scores(bm25, q["question"])
        d = cvec @ qvec[qi]  # cosine; vectors are L2-normalised (== FAISS IndexFlatIP)
        border = list(np.argsort(-b, kind="stable"))
        dorder = list(np.argsort(-d, kind="stable"))
        rankings["bm25"].append(border)
        rankings["dense"].append(dorder)
        # hybrid: each retriever contributes its top 2k, k = CANDIDATES (as HybridRetriever does)
        n = 2 * CANDIDATES
        fused = fuse({int(i): float(d[i]) for i in dorder[:n]},
                     {int(i): float(b[i]) for i in border[:n]}, SEM_W, BM25_W)
        rankings["hybrid"].append(fused)
        cand = fused[:CANDIDATES]
        rs = reranker.predict([(q["question"], chunks[i][3]) for i in cand], batch_size=32)
        order = [cand[j] for j in np.argsort(-np.asarray(rs), kind="stable")]
        rankings["hybrid+rerank"].append(order)
    return rankings


def score(chunks, questions, golds, rankings):
    spans = [c[:3] for c in chunks]
    per = {m: {"qid": [], "type": [], **{f"recall@{k}": [] for k in KS},
               "mrr": [], "ndcg@10": []} for m in METHODS}
    for m in METHODS:
        for q, g, order in zip(questions, golds, rankings[m]):
            ranked = [spans[i] for i in order]
            n_rel = sum(any(overlaps(s, x) for x in g) for s in spans)
            p = per[m]
            p["qid"].append(q["id"]); p["type"].append(q["type"])
            for k in KS:
                p[f"recall@{k}"].append(recall_at_k(ranked, g, k))
            p["mrr"].append(reciprocal_rank(ranked, g))
            p["ndcg@10"].append(ndcg_at_k(ranked, g, 10, n_rel))
    return per


def summarise(per):
    metrics = [f"recall@{k}" for k in KS] + ["mrr", "ndcg@10"]
    out = {}
    for m, p in per.items():
        out[m] = {"overall": {}, "by_type": {}}
        for name in metrics:
            mean, lo, hi = bootstrap_ci(p[name])
            out[m]["overall"][name] = {"mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        for t in sorted(set(p["type"])):
            idx = [i for i, x in enumerate(p["type"]) if x == t]
            out[m]["by_type"][t] = {"n": len(idx), **{
                name: round(float(np.mean([p[name][i] for i in idx])), 4) for name in metrics}}
    return out


def markdown(res):
    lines = []
    for ch, summary in res["chunkers"].items():
        lines += [f"### Chunking: `{ch}` ({summary['n_chunks']} chunks)", "",
                  "| Method | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |",
                  "|---|---|---|---|---|---|"]
        for m in METHODS:
            o = summary["results"][m]["overall"]
            cell = lambda k: f"{o[k]['mean']:.3f} [{o[k]['ci95'][0]:.3f}, {o[k]['ci95'][1]:.3f}]"
            lines.append(f"| {m} | " + " | ".join(
                cell(k) for k in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10")) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gold", default=str(ROOT / "data/eval/retrieval_gold.json"))
    ap.add_argument("--embedding-backend", default="sentence-transformers",
                    choices=["sentence-transformers", "openai"])
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--date", default=date.today().isoformat())
    args = ap.parse_args()

    gold = json.loads(Path(args.gold).read_text())
    corpus_dir = ROOT / gold["corpus_dir"]
    docs = {p.name: p.read_text() for p in sorted(corpus_dir.glob("*.txt"))}
    answerable = [q for q in gold["questions"] if q["evidence"]]
    golds = [[(e["doc"], *resolve_quote(docs[e["doc"]], e["quote"])) for e in q["evidence"]]
             for q in answerable]

    from sentence_transformers import CrossEncoder
    embed = make_embedder(args.embedding_backend, args.embedding_model)
    reranker = CrossEncoder(args.rerank_model)

    res = {"date": args.date, "gold_file": str(Path(args.gold).relative_to(ROOT)) if Path(args.gold).is_relative_to(ROOT) else args.gold,
           "n_questions_total": len(gold["questions"]),
           "n_questions_evaluated": len(answerable),
           "n_not_covered_excluded": len(gold["questions"]) - len(answerable),
           "n_documents": len(docs),
           "embedding_backend": args.embedding_backend, "embedding_model": args.embedding_model,
           "rerank_model": args.rerank_model, "hybrid_weights": {"semantic": SEM_W, "bm25": BM25_W},
           "rerank_candidates": CANDIDATES, "bootstrap": {"n_boot": 2000, "seed": 0, "level": 0.95},
           "environment": {"python": platform.python_version(), "machine": platform.machine(),
                           "platform": platform.platform()},
           "chunkers": {}}
    for name, fn in CHUNKERS.items():
        chunks = [c for d, t in docs.items() for c in fn(d, t)]
        per = score(chunks, answerable, golds, rank_all(chunks, embed, reranker, answerable))
        res["chunkers"][name] = {"n_chunks": len(chunks), "results": summarise(per)}
        print(f"chunker={name}: {len(chunks)} chunks", file=sys.stderr)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"retrieval_eval_{args.date}.json").write_text(json.dumps(res, indent=2) + "\n")
    (out / f"retrieval_eval_{args.date}.md").write_text(markdown(res) + "\n")
    print(markdown(res))


if __name__ == "__main__":
    main()

"""Command line: ``python -m src.impact.cli impact|diff|ask``."""

from __future__ import annotations

import argparse
import json
import sys

from .analysis import (
    ImpactConfig,
    analyze_impact,
    evaluate_against_gold,
    load_corpus,
    load_questions,
    render_markdown,
)
from .answer import answer_question, generator_from_env
from .diff import diff_corpus
from .embedders import CrossEncoderReranker, make_embedder
from .index import ClauseIndex
from .settings import Settings


def main(argv: list[str] | None = None) -> int:
    s = Settings()
    p = argparse.ArgumentParser(prog="polirag-impact", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--backend", default=s.embedding_backend, choices=["hashing", "sentence-transformers"]
        )
        sp.add_argument("--model", default=s.embedding_model)
        sp.add_argument("--rerank", action="store_true", default=s.rerank_enabled)
        sp.add_argument("--k", type=int, default=2)

    i = sub.add_parser("impact", help="which answers are affected by a policy revision")
    i.add_argument("--old", default=str(s.v1_dir), help="directory of v1 *.txt")
    i.add_argument("--new", default=str(s.v2_dir), help="directory of v2 *.txt")
    i.add_argument("--gold", default=str(s.gold_path), help="questions (retrieval_gold.json)")
    i.add_argument("--json", dest="json_out", help="write the full JSON report here")
    i.add_argument("--min-severity", default="medium", choices=["low", "medium", "high"])
    i.add_argument("--no-refine", action="store_true")
    i.add_argument(
        "--score-vs-gold",
        action="store_true",
        help="also report precision/recall against the gold-evidence oracle",
    )
    common(i)
    d = sub.add_parser("diff", help="clause-level diff of two corpora")
    d.add_argument("--old", default=str(s.v1_dir))
    d.add_argument("--new", default=str(s.v2_dir))
    a = sub.add_parser("ask", help="answer a question with clause citations")
    a.add_argument("question")
    a.add_argument("--corpus", default=str(s.v1_dir))
    common(a)
    args = p.parse_args(argv)

    if args.cmd == "diff":
        for c in diff_corpus(load_corpus(args.old).docs, load_corpus(args.new).docs):
            if c.kind != "unchanged":
                print(f"{c.kind:10s} {(c.new or c.old).id}  {c.word_changes[:2]}")  # type: ignore[union-attr]
        return 0
    emb = make_embedder(args.backend, args.model)
    rer = CrossEncoderReranker(s.rerank_model) if args.rerank else None
    if args.cmd == "ask":
        hits = ClauseIndex(load_corpus(args.corpus).docs, emb, rer).search(args.question, args.k)
        print(
            json.dumps(
                answer_question(args.question, hits, generator_from_env()).to_dict(), indent=2
            )
        )
        return 0
    old, new, qs = load_corpus(args.old), load_corpus(args.new), load_questions(args.gold)
    cfg = ImpactConfig(k=args.k, min_severity=args.min_severity, refine=not args.no_refine)
    report = analyze_impact(old, new, qs, emb, rer, cfg, generator_from_env())
    if args.score_vs_gold:
        report["gold_oracle_eval"] = evaluate_against_gold(report, old, new, qs)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    sys.stdout.write(render_markdown(report))
    if "gold_oracle_eval" in report:
        print("\nGold-oracle evaluation:", json.dumps(report["gold_oracle_eval"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

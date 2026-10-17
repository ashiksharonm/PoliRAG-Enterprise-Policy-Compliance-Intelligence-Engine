# Impact-analysis evaluation (2026-10-17)

Setup: `data/raw` (v1) vs `data/versions/v2` (v2, built by `scripts/make_v2_corpus.py`: 5 modified clauses,
1 added, 1 removed; all edits are **synthetic**, the ACME policies are fictional), the 76-question
retrieval gold set, Apple M5 16 GB, CPU. Embedding `all-MiniLM-L6-v2`, hybrid BM25 + dense, cross-encoder
rerank `ms-marco-MiniLM-L-6-v2`, extractive answers, refinement on, severity threshold `medium`.

Ground truth (oracle): an answerable question is "truly affected" when one of its gold evidence quotes is no
longer present verbatim in the v2 document. 68 questions are answerable; 6 are truly affected.
The 8 `not_covered` questions have no oracle and are excluded from precision/recall.

| evidence depth k | flagged | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|---|
| 1 | 8 | 6 | 1 | 0 | 0.857 | 1.000 |
| 2 (default) | 11 | 6 | 4 | 0 | 0.600 | 1.000 |
| 3 | 12 | 6 | 5 | 0 | 0.545 | 1.000 |

Without model downloads (`--backend hashing`, lexical hashing embedder, no rerank): k=1 P 0.833 / R 0.833,
k=2 P 0.556 / R 0.833, k=3 P 0.714 / R 0.833.

Reproduce: `python -m src.impact.cli impact --backend sentence-transformers --rerank --k 2 --score-vs-gold`.
Full per-question reports: `impact_k1.json`, `impact_k2.json`, `impact_k3.json`.

Caveats
- Only 6 truly affected questions: precision and recall have very wide uncertainty; treat as a sanity check.
- Most "false positives" are questions whose retrieved evidence sits in a changed clause but whose gold phrase
  is untouched (for example a gift-card question retrieving the edited gift-value clause). They are
  conservative flags for a reviewer, not errors in diffing.
- Unanswerable questions can still be flagged when retrieval returns an unrelated changed clause (q072,
  "salary review", retrieves the removed vendor clause). There is no abstention threshold yet.
- Edits are author-written and the gold set is single-annotator; results do not transfer to real redlines.

# Evaluation data

## `retrieval_gold.json` (retrieval gold set, v1.0)

76 questions over the 7 policy documents in `data/raw/`:

| Type | n | Notes |
|---|---|---|
| factual | 20 | single-clause lookup |
| numeric | 17 | thresholds, durations, limits |
| multi_hop | 12 | answer needs evidence from 2 passages (usually 2 documents) |
| paraphrase | 11 | wording deliberately differs from the policy text |
| negation | 8 | yes/no questions whose answer is "no / prohibited" |
| not_covered | 8 | topic absent from the corpus; no evidence, excluded from retrieval metrics |

### How it was built

- **Single annotator, author-written.** Questions, gold answers and evidence were
  written by one person (the repo owner) by reading the corpus. There is no
  inter-annotator agreement and no external validation.
- **Corpus.** `sample_policy.txt` (POL-2025-001) was the only document in the repo.
  Multi-hop questions need more than one document, so six short synthetic policy
  documents (POL-2025-002 to -007) were written for this purpose. The fictional ACME
  company and all figures are invented. Several topics (retention periods, breach
  notification windows, encryption) deliberately recur across documents so the
  retrievers face near-duplicate distractors.
- **Chunking-independent gold.** Each evidence item is `{doc, quote}`, a verbatim
  phrase from the document (whitespace-insensitive). The scorer resolves it to a
  character span, and a retrieved chunk is relevant if it overlaps that span, so
  the same set scores any chunking strategy. `tests/test_retrieval_metrics.py`
  checks that every quote occurs exactly once in its document.
- **Metrics.** Recall@k is the fraction of a question's evidence spans covered by
  the top-k chunks (so a 2-span multi-hop question has Recall@1 <= 0.5). MRR uses
  the first overlapping chunk. nDCG@10 uses binary gains.
- **Known bias.** Questions were written with the text in view, so lexical overlap
  with the source is likely higher than for real user queries, except for the
  paraphrase subset, which was written to reduce it.

## `golden_qa.json`

The original 5-question smoke test used by `scripts/evaluate.py`. It has no gold
answers or evidence and is not used for the retrieval results.

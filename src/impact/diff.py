"""Clause-level diff between two versions of a policy corpus."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from .clauses import Clause, PolicyDoc

MATCH_THRESHOLD = 0.6  # minimum similarity to call two clauses "the same clause, edited"


@dataclass
class ClauseChange:
    kind: str  # unchanged | modified | renumbered | added | removed
    old: Clause | None
    new: Clause | None
    similarity: float = 1.0
    word_changes: list[tuple[str, str]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.kind in ("modified", "added", "removed")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "old_id": self.old.id if self.old else None,
            "new_id": self.new.id if self.new else None,
            "title": ref.title if (ref := self.new or self.old) else None,
            "similarity": round(self.similarity, 3),
            "word_changes": [{"old": a, "new": b} for a, b in self.word_changes],
        }


def _tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def similarity(a: Clause, b: Clause) -> float:
    return difflib.SequenceMatcher(None, _tokens(a.text), _tokens(b.text), autojunk=False).ratio()


def word_changes(a: Clause, b: Clause, limit: int = 12) -> list[tuple[str, str]]:
    """Replaced / inserted / deleted word spans, e.g. ``("30 calendar days", "45 calendar days")``."""
    ta, tb = _tokens(a.body), _tokens(b.body)
    out: list[tuple[str, str]] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        if op != "equal":
            ctx_a = " ".join(ta[max(i1 - 1, 0) : i2 + 1])
            ctx_b = " ".join(tb[max(j1 - 1, 0) : j2 + 1])
            out.append((ctx_a if i2 > i1 else "", ctx_b if j2 > j1 else ""))
    return out[:limit]


def diff_docs(old: PolicyDoc, new: PolicyDoc) -> list[ClauseChange]:
    """Match clauses of two versions of one document and classify each pair.

    Order of matching: same number and title, then identical content under a new number
    (renumbered), then best text similarity above ``MATCH_THRESHOLD`` (modified).
    """
    changes: list[ClauseChange] = []
    old_left = {c.id: c for c in old.clauses}
    new_left = {c.id: c for c in new.clauses}

    def settle(o: Clause, n: Clause) -> None:
        if o.content_hash == n.content_hash:
            kind = "unchanged" if o.number == n.number else "renumbered"
            changes.append(ClauseChange(kind, o, n))
        else:
            changes.append(ClauseChange("modified", o, n, similarity(o, n), word_changes(o, n)))
        old_left.pop(o.id, None)
        new_left.pop(n.id, None)

    for o in list(old_left.values()):  # same number + same title
        n = next(
            (
                x
                for x in new_left.values()
                if x.number == o.number and x.title.lower() == o.title.lower()
            ),
            None,
        )
        if n:
            settle(o, n)
    for o in list(old_left.values()):  # identical content, new number
        n = next((x for x in new_left.values() if x.content_hash == o.content_hash), None)
        if n:
            settle(o, n)
    pairs = sorted(
        ((similarity(o, n), o.id, n.id) for o in old_left.values() for n in new_left.values()),
        reverse=True,
    )
    for sim, oid, nid in pairs:  # greedy best-first on the remainder
        if sim < MATCH_THRESHOLD:
            break
        if oid in old_left and nid in new_left:
            settle(old_left[oid], new_left[nid])
    changes += [ClauseChange("removed", o, None, 0.0) for o in old_left.values()]
    changes += [ClauseChange("added", None, n, 0.0) for n in new_left.values()]
    return changes


def diff_corpus(old: list[PolicyDoc], new: list[PolicyDoc]) -> list[ClauseChange]:
    """Diff every document present in either corpus (documents are paired by Document ID)."""
    o, n = {d.doc_id: d for d in old}, {d.doc_id: d for d in new}
    out: list[ClauseChange] = []
    for doc_id in sorted(set(o) | set(n)):
        out += diff_docs(
            o.get(doc_id, PolicyDoc(doc_id, doc_id, None)),
            n.get(doc_id, PolicyDoc(doc_id, doc_id, None)),
        )
    return out

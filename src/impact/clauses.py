"""Parse numbered policy documents into addressable clauses.

Format handled (as in ``data/raw``): a title line, ``Document ID: X``, numbered sections
(``3. RETENTION PERIODS``) and numbered clauses (``3.1 Financial Records``) followed by body text.
A section with no numbered sub-clauses (``1. PURPOSE``) is itself a clause.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_SECTION = re.compile(r"^(\d+)\.\s+([A-Z][A-Za-z ,&/'()-]*)$")
_CLAUSE = re.compile(r"^(\d+\.\d+[a-z]?)\s+(\S.*)$")
_DOC_ID = re.compile(r"^Document ID:[ \t]*(\S+)", re.M)
_EFFECTIVE = re.compile(r"^Effective Date:[ \t]*(\S.*)$", re.M)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class Clause:
    doc_id: str
    number: str
    title: str
    body: str

    @property
    def id(self) -> str:
        """Stable citation id, e.g. ``POL-2025-005#3.2``."""
        return f"{self.doc_id}#{self.number}"

    @property
    def text(self) -> str:
        """Retrieval text: number, title and body on normalised whitespace."""
        return _norm(f"{self.number} {self.title}. {self.body}")

    @property
    def content_hash(self) -> str:
        """Hash of the title and body only, so a pure renumbering keeps the same hash."""
        return hashlib.sha256(_norm(f"{self.title}. {self.body}").lower().encode()).hexdigest()[:16]


@dataclass
class PolicyDoc:
    doc_id: str
    title: str
    effective_date: str | None
    clauses: list[Clause] = field(default_factory=list)

    def by_id(self) -> dict[str, Clause]:
        return {c.id: c for c in self.clauses}


def parse_policy(text: str, name: str = "document") -> PolicyDoc:
    """Split ``text`` into clauses. ``name`` is used as the id when no Document ID line exists."""
    m = _DOC_ID.search(text)
    doc_id = m.group(1) if m else re.sub(r"\W+", "-", name).strip("-") or "document"
    eff = _EFFECTIVE.search(text)
    lines = text.splitlines()
    title = next((ln.strip() for ln in lines if ln.strip()), name)
    clauses: list[Clause] = []
    cur: tuple[str, str] | None = None
    body: list[str] = []
    section: tuple[str, str] | None = None
    section_body: list[str] = []
    section_has_sub = False

    def flush_clause() -> None:
        nonlocal cur, body
        if cur is not None:
            clauses.append(Clause(doc_id, cur[0], cur[1], _norm(" ".join(body))))
        cur, body = None, []

    def flush_section() -> None:
        nonlocal section, section_body
        if section is not None and not section_has_sub and _norm(" ".join(section_body)):
            clauses.append(
                Clause(doc_id, section[0], section[1].title(), _norm(" ".join(section_body)))
            )
        section, section_body = None, []

    for raw in lines:
        line = raw.strip()
        sm, cm = _SECTION.match(line), _CLAUSE.match(line)
        if sm:
            flush_clause()
            flush_section()
            section, section_has_sub = (sm.group(1), sm.group(2).strip()), False
        elif cm:
            flush_clause()
            cur, section_has_sub = (cm.group(1), cm.group(2).strip()), True
        elif cur is not None:
            body.append(line)
        elif section is not None:
            section_body.append(line)
    flush_clause()
    flush_section()
    return PolicyDoc(doc_id, title, eff.group(1).strip() if eff else None, clauses)

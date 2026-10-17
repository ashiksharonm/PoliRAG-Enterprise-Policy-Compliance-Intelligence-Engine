"""Answers that cite clause ids: extractive fallback and optional free-tier LLM generation."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import httpx

from .index import Hit

_STOP = set(
    "a an and are as at be by can do does for from has have how in is it its of on or "
    "that the their to under what when which who will with within must may shall".split()
)
_CITE = re.compile(r"\[([A-Za-z0-9._-]+#[0-9.a-z]+)\]")


@dataclass
class Answer:
    text: str
    citations: list[str]
    mode: str  # extractive | groq | gemini | extractive-fallback

    def to_dict(self) -> dict[str, object]:
        return {"text": self.text, "citations": self.citations, "mode": self.mode}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP}


def extractive_answer(question: str, hits: list[Hit], max_sentences: int = 2) -> Answer:
    """Pick the sentences from the retrieved clauses that overlap the question most.

    No generation, so nothing can be said that is not in a cited clause.
    """
    if not hits:
        return Answer("No relevant clause found.", [], "extractive")
    q = _words(question)
    scored: list[tuple[float, int, int, str, str, float]] = []
    for h in hits:
        for si, sent in enumerate(re.split(r"(?<=[.;])\s+", h.clause.body)):
            if sent.strip():
                overlap = len(q & _words(sent)) / (1 + 0.05 * len(sent.split()))
                scored.append(
                    (
                        overlap + (0.5 if h.rank == 1 else 0.0),
                        h.rank,
                        si,
                        sent.strip(),
                        h.clause.id,
                        overlap,
                    )
                )
    ranked = sorted(scored, key=lambda t: (-t[0], t[1], t[2]))
    best = ([t for t in ranked if t[5] > 0] or ranked)[:max_sentences]
    best.sort(key=lambda t: (t[1], t[2]))
    cites: list[str] = []
    for t in best:
        cid = t[4]
        if cid not in cites:
            cites.append(cid)
    return Answer(" ".join(f"{t[3]} [{t[4]}]" for t in best), cites, "extractive")


def build_prompt(question: str, hits: list[Hit]) -> str:
    ctx = "\n\n".join(f"[{h.clause.id}] {h.clause.title}\n{h.clause.body}" for h in hits)
    return (
        "Answer the question using ONLY the policy clauses below. Cite every claim with the "
        "clause id in square brackets, e.g. [POL-2025-001#3.1]. If the clauses do not answer "
        "the question, reply exactly: Not covered by the provided clauses.\n\n"
        f"Clauses:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    )


def validate_citations(text: str, hits: list[Hit]) -> list[str] | None:
    """Cited ids, or None when the answer cites nothing or cites an id that was not provided."""
    cited = _CITE.findall(text)
    allowed = {h.clause.id for h in hits}
    if "Not covered" in text and not cited:
        return []
    if not cited or any(c not in allowed for c in cited):
        return None
    return list(dict.fromkeys(cited))


class LLMGenerator:
    """Groq or Gemini free tier over plain HTTPS. Keys come from the environment."""

    def __init__(self, provider: str, api_key: str, model: str, timeout: float = 20.0) -> None:
        if provider not in ("groq", "gemini"):
            raise ValueError("provider must be groq or gemini")
        self.provider, self.api_key, self.model, self.timeout = provider, api_key, model, timeout

    def complete(self, prompt: str) -> str:
        if self.provider == "groq":
            r = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
                json={
                    "model": self.model,
                    "temperature": 0,
                    "max_tokens": 300,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            r.raise_for_status()
            return str(r.json()["choices"][0]["message"]["content"]).strip()
        r = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            headers={"x-goog-api-key": self.api_key},
            timeout=self.timeout,
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": 300},
            },
        )
        r.raise_for_status()
        return str(r.json()["candidates"][0]["content"]["parts"][0]["text"]).strip()


def generator_from_env() -> LLMGenerator | None:
    """Groq if GROQ_API_KEY is set, else Gemini if GEMINI_API_KEY is set, else None."""
    if key := os.environ.get("GROQ_API_KEY"):
        return LLMGenerator("groq", key, os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"))
    if key := os.environ.get("GEMINI_API_KEY"):
        return LLMGenerator("gemini", key, os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))
    return None


def answer_question(
    question: str, hits: list[Hit], generator: LLMGenerator | None = None
) -> Answer:
    """LLM answer when a generator is configured and its citations validate, else extractive."""
    if generator is None or not hits:
        return extractive_answer(question, hits)
    try:
        text = generator.complete(build_prompt(question, hits))
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        return Answer(
            **{**extractive_answer(question, hits).__dict__, "mode": "extractive-fallback"}
        )
    cites = validate_citations(text, hits)
    if cites is None:
        return Answer(
            **{**extractive_answer(question, hits).__dict__, "mode": "extractive-fallback"}
        )
    return Answer(text, cites, generator.provider)

"""FastAPI service for policy change impact analysis.

uvicorn src.impact.service:create_app --factory --port 8080
"""

from __future__ import annotations

import time
import uuid
from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from . import __version__
from .analysis import (
    Corpus,
    ImpactConfig,
    Question,
    analyze_impact,
    corpus_from_texts,
    load_corpus,
    load_questions,
)
from .answer import answer_question, generator_from_env
from .diff import diff_corpus
from .embedders import CrossEncoderReranker, Embedder, make_embedder
from .index import ClauseIndex, Reranker
from .logging_config import configure, request_id
from .settings import Settings


class CorpusRef(BaseModel):
    """Either a bundled version name (``v1``, ``v2``) or raw policy files {name: text}."""

    version: str | None = None
    files: dict[str, str] | None = None

    @model_validator(mode="after")
    def _one_of(self) -> CorpusRef:
        if (self.version is None) == (self.files is None):
            raise ValueError("give exactly one of 'version' or 'files'")
        return self


class QuestionIn(BaseModel):
    id: str | None = None
    text: str = Field(min_length=3, max_length=500)


class ImpactRequest(BaseModel):
    old: CorpusRef
    new: CorpusRef
    questions: list[QuestionIn] | None = Field(None, description="default: bundled gold set")
    k: int = Field(2, ge=1, le=5)
    min_severity: str = Field("medium", pattern="^(low|medium|high)$")
    refine: bool = True


class DiffRequest(BaseModel):
    old: CorpusRef
    new: CorpusRef


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    corpus: CorpusRef = CorpusRef(version="v1")
    k: int = Field(3, ge=1, le=5)


class State:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._embedder: Embedder | None = None
        self._reranker: Reranker | None = None

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            s = self.settings
            self._embedder = make_embedder(s.embedding_backend, s.embedding_model)
        return self._embedder

    @property
    def reranker(self) -> Reranker | None:
        if self.settings.rerank_enabled and self._reranker is None:
            self._reranker = CrossEncoderReranker(self.settings.rerank_model)
        return self._reranker

    @lru_cache(maxsize=4)  # noqa: B019 - one State per app
    def bundled(self, version: str) -> Corpus:
        paths = {"v1": self.settings.v1_dir, "v2": self.settings.v2_dir}
        if version not in paths:
            raise HTTPException(404, f"unknown version {version!r}; available: v1, v2")
        return load_corpus(paths[version])

    def resolve(self, ref: CorpusRef) -> Corpus:
        if ref.version is not None:
            return self.bundled(ref.version)
        assert ref.files is not None
        if (
            not ref.files
            or sum(len(t) for t in ref.files.values()) > self.settings.max_request_bytes
        ):
            raise HTTPException(413, "files empty or larger than MAX_REQUEST_BYTES")
        return corpus_from_texts(ref.files)


def create_app(settings: Settings | None = None) -> FastAPI:
    st = State(settings or Settings())
    log = configure(st.settings.log_level)
    app = FastAPI(title="PoliRAG policy change impact", version=__version__)

    def auth(x_api_key: str | None = Header(None)) -> None:
        if st.settings.api_key and x_api_key != st.settings.api_key:
            raise HTTPException(401, "invalid or missing X-API-Key")

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Any:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id.set(rid)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            ms = round((time.perf_counter() - t0) * 1000, 1)
            log.info("request", extra={"extra_fields": {"path": request.url.path, "ms": ms}})
            request_id.reset(token)
        response.headers["x-request-id"] = rid
        return response

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        s = st.settings
        ok = s.v1_dir.is_dir() and s.v2_dir.is_dir()
        if not ok:
            raise HTTPException(503, "bundled corpora not found")
        return {
            "status": "ready",
            "embedding_backend": s.embedding_backend,
            "embedding_model": s.embedding_model if s.embedding_backend != "hashing" else None,
        }

    @app.get("/v1/versions", dependencies=[Depends(auth)])
    def versions() -> dict[str, Any]:
        out = {}
        for v in ("v1", "v2"):
            c = st.bundled(v)
            out[v] = {
                "documents": len(c.docs),
                "clauses": sum(len(d.clauses) for d in c.docs),
                "files": sorted(c.files),
            }
        return out

    @app.post("/v1/diff", dependencies=[Depends(auth)])
    def diff(req: DiffRequest) -> dict[str, Any]:
        ch = diff_corpus(st.resolve(req.old).docs, st.resolve(req.new).docs)
        return {
            "changes": [c.to_dict() for c in ch if c.kind != "unchanged"],
            "counts": {
                k: sum(c.kind == k for c in ch)
                for k in ("unchanged", "renumbered", "modified", "added", "removed")
            },
        }

    @app.post("/v1/impact", dependencies=[Depends(auth)])
    def impact(req: ImpactRequest) -> dict[str, Any]:
        old, new = st.resolve(req.old), st.resolve(req.new)
        if req.questions is not None:
            qs = [Question(q.id or f"q{i + 1}", q.text) for i, q in enumerate(req.questions)]
        else:
            qs = load_questions(st.settings.gold_path)
        if len(qs) > st.settings.max_questions:
            raise HTTPException(413, f"at most {st.settings.max_questions} questions")
        cfg = ImpactConfig(k=req.k, min_severity=req.min_severity, refine=req.refine)
        return analyze_impact(old, new, qs, st.embedder, st.reranker, cfg, generator_from_env())

    @app.post("/v1/ask", dependencies=[Depends(auth)])
    def ask(req: AskRequest) -> dict[str, Any]:
        corpus = st.resolve(req.corpus)
        hits = ClauseIndex(corpus.docs, st.embedder, st.reranker).search(req.question, req.k)
        ans = answer_question(req.question, hits, generator_from_env())
        return {
            "answer": ans.to_dict(),
            "evidence": [
                {
                    "clause_id": h.clause.id,
                    "title": h.clause.title,
                    "rank": h.rank,
                    "text": h.clause.body,
                }
                for h in hits
            ],
        }

    return app

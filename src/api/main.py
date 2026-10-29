"""API REST del asistente (FastAPI). Se arranca con `python scripts/run_api.py`."""

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.analytics import db
from src.api import stats
from src.api.schemas import AskRequest, AskResponse, SearchRequest, SearchResponse, Source
from src.generation.ollama import DEFAULT_MODEL, OllamaClient, OllamaUnavailable
from src.generation.rag import REFUSAL, RAGPipeline
from src.retrieval.service import PROCESSED, build_retriever

# Modelos que no caben en la VRAM de la GPU de desarrollo (4 GB) y deben correr solo en CPU.
CPU_ONLY_MODELS = {m.strip() for m in os.environ.get("RAG_CPU_ONLY_MODELS", "qwen2.5:7b-instruct").split(",") if m.strip()}


def default_llm_factory(model: str) -> OllamaClient:
    return OllamaClient(model, num_gpu=0 if model in CPU_ONLY_MODELS else None)


def _sources(hits) -> list[Source]:
    return [Source(n=i, doc_id=h.doc_id, fuente=h.metadata.get("fuente"),
                   pagina=None if h.metadata.get("pagina") is None else int(h.metadata["pagina"]),
                   doc_type=h.metadata.get("doc_type"), score=round(h.rerank_score if h.rerank_score is not None
                                                                     else h.score, 4), text=h.text)
            for i, h in enumerate(hits, start=1)]


def create_app(retriever=None, llm_factory=None, con=None, default_model: str | None = None,
               processed_dir: Path = PROCESSED, years=(2025, 2026)) -> FastAPI:
    llm_factory = llm_factory or default_llm_factory
    default_model = default_model or os.environ.get("RAG_MODEL", DEFAULT_MODEL)
    state: dict = {"retriever": retriever, "con": con, "pipelines": {}}
    ask_lock = threading.Semaphore(1)  # un LLM a la vez: evita saturar GPU/CPU

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["retriever"] is None:
            state["retriever"] = build_retriever(processed_dir, rerank=os.environ.get("RAG_RERANK") == "1")
        if state["con"] is None:
            state["con"] = db.connect(processed_dir, years=list(years))
        yield

    app = FastAPI(title="Asistente del mercado automotriz de Ecuador", version="1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    if retriever is not None and con is not None:  # inyectados (tests): sin lifespan
        state["retriever"], state["con"] = retriever, con

    def installed_models() -> list[str]:
        try:
            return llm_factory(default_model).installed_models()
        except OllamaUnavailable:
            return []

    def pipeline_for(model: str, k: int) -> RAGPipeline:
        key = (model, k)
        if key not in state["pipelines"]:
            state["pipelines"][key] = RAGPipeline(state["retriever"], llm_factory(model), k=k)
        return state["pipelines"][key]

    @app.get("/health")
    def health():
        models = installed_models()
        return {"status": "ok", "corpus_docs": len(state["retriever"].corpus), "ollama": bool(models),
                "models": models, "default_model": default_model}

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest):
        r = state["retriever"]
        hits = r.search(req.query, k=req.k, mode=req.mode, understand=req.understand, rerank=req.rerank,
                        filters=req.filters)
        return SearchResponse(query=req.query, hits=_sources(hits), parsed_filters=r.parser.parse(req.query))

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest):
        model = req.model or default_model
        t0 = time.time()
        with ask_lock:
            try:
                installed = installed_models()
                if installed and model not in installed:
                    raise HTTPException(400, f"Modelo no instalado: {model}. Disponibles: {', '.join(installed)}")
                ans = pipeline_for(model, req.k).answer(req.question)
            except OllamaUnavailable as exc:
                raise HTTPException(503, str(exc)) from exc
        return AskResponse(
            question=req.question, answer=ans.text, grounded=ans.verdict.grounded,
            refused=ans.text.startswith(REFUSAL), retried=ans.retried, problems=ans.verdict.problems,
            filters=ans.filters, sources=_sources(ans.hits), model=model, seconds=round(time.time() - t0, 1))

    @app.get("/stats/meta")
    def stats_meta():
        return stats.meta(state["con"])

    @app.get("/stats/kpis")
    def stats_kpis():
        return stats.kpis(state["con"])

    @app.get("/stats/mensual")
    def stats_mensual(marca: str | None = None, provincia: str | None = None, combustible: str | None = None,
                      clase: str | None = None):
        return stats.monthly(state["con"], marca=marca, provincia=provincia, combustible=combustible, clase=clase)

    @app.get("/stats/top")
    def stats_top(dimension: str, anio: int, limit: int = Query(10, ge=1, le=30), mes_hasta: int | None = Query(None, ge=1, le=12),
                  marca: str | None = None, provincia: str | None = None, combustible: str | None = None,
                  clase: str | None = None):
        try:
            return stats.top(state["con"], dimension, anio, limit, mes_hasta, marca=marca, provincia=provincia,
                             combustible=combustible, clase=clase)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    return app


app = create_app()

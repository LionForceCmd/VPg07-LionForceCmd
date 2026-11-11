"""generation_pipeline: вопрос -> Pinecone (документы + память) -> агент -> ответ

    text_embedder ─embedding─> doc_retriever ──documents──┐
                  └─embedding─> mem_retriever ──memories──┴> context_builder ──messages──> agent

Агент (Haystack Agent) сохраняет все инструменты v1 (кошки, собаки, погода) и
получает найденные фрагменты документов и прошлые сообщения как контекст.
"""
from __future__ import annotations

from haystack import Pipeline
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever

from components.context_builder import ContextBuilder
from components.context import DOC_TYPE, MESSAGE_TYPE, type_user_filter
from components.llm import build_agent
from components.context import build_text_embedder


def build_generation_pipeline(document_store, text_embedder=None, agent=None) -> Pipeline:
    p = Pipeline()
    p.add_component("text_embedder", text_embedder or build_text_embedder())
    p.add_component("doc_retriever", PineconeEmbeddingRetriever(document_store=document_store))
    p.add_component("mem_retriever", PineconeEmbeddingRetriever(document_store=document_store))
    p.add_component("context_builder", ContextBuilder())
    p.add_component("agent", agent or build_agent())

    p.connect("text_embedder.embedding", "doc_retriever.query_embedding")
    p.connect("text_embedder.embedding", "mem_retriever.query_embedding")
    p.connect("doc_retriever.documents", "context_builder.documents")
    p.connect("mem_retriever.documents", "context_builder.memories")
    p.connect("context_builder.messages", "agent.messages")
    return p


def run_generation(pipeline: Pipeline, user_id: int, text: str, history: list,
                   docs_top_k: int, mem_top_k: int) -> tuple[str, list]:
    """Возвращает (ответ, найденные фрагменты документов)."""
    out = pipeline.run(
        {
            "text_embedder": {"text": text},
            "doc_retriever": {"filters": type_user_filter(DOC_TYPE, user_id), "top_k": docs_top_k},
            "mem_retriever": {"filters": type_user_filter(MESSAGE_TYPE, user_id), "top_k": mem_top_k},
            "context_builder": {"query": text, "history": history},
        },
        include_outputs_from={"doc_retriever"},
    )
    answer = out["agent"]["last_message"].text or "…"
    return answer, out.get("doc_retriever", {}).get("documents", [])

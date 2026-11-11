"""ingestion_pipeline: файл -> Docling (разбор + чанки) -> эмбеддинги -> Pinecone
                                                     \\-> выборка текста -> LLM -> резюме

    chunker ──documents──> embedder ──documents──> writer        (запись в Pinecone)
       └─────documents──> sampler ─text/file_name─> prompt ──> llm   (резюме в 1 предложение)
"""
from __future__ import annotations

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy

import config
from components.docling_chunker import DoclingChunker
from components.llm import build_chat_generator, summary_template
from components.text_sampler import TextSampler


def build_ingestion_pipeline(document_store, doc_embedder, chat_generator=None, chunker=None) -> Pipeline:
    p = Pipeline()
    p.add_component("chunker", chunker or DoclingChunker(
        max_tokens=config.CHUNK_MAX_TOKENS, do_ocr=config.DOCLING_DO_OCR))
    p.add_component("embedder", doc_embedder)
    p.add_component("writer", DocumentWriter(document_store, policy=DuplicatePolicy.OVERWRITE))
    p.add_component("sampler", TextSampler(max_chars=config.SUMMARY_MAX_CHARS))
    p.add_component("prompt", ChatPromptBuilder(
        template=summary_template(), required_variables=["file_name", "text"]))
    p.add_component("llm", chat_generator or build_chat_generator())

    p.connect("chunker.documents", "embedder.documents")
    p.connect("embedder.documents", "writer.documents")
    p.connect("chunker.documents", "sampler.documents")
    p.connect("sampler.text", "prompt.text")
    p.connect("sampler.file_name", "prompt.file_name")
    p.connect("prompt.prompt", "llm.messages")
    return p


def run_ingestion(pipeline: Pipeline, path: str, file_name: str, user_id: int) -> tuple[int, str]:
    """Запускает пайплайн. Возвращает (сколько чанков записано, резюме одним предложением)."""
    out = pipeline.run({"chunker": {"sources": [path], "file_names": [file_name], "user_id": user_id}})
    written = out.get("writer", {}).get("documents_written", 0)
    replies = out.get("llm", {}).get("replies") or []
    summary = (replies[0].text or "").strip() if replies else ""
    return written, summary

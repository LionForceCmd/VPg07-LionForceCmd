"""Долговременная память: фабрики Pinecone/эмбеддеров и ContextStore.

В одном индексе Pinecone живут два типа записей (различаются ``meta.type``):
* ``user_message``  -- сообщения пользователя (память диалога, как в v1);
* ``document_chunk`` -- чанки загруженных файлов (их пишет ingestion_pipeline).

Эмбеддинги считает ТОЛЬКО OpenAI-совместимый API по ``OPENAI_BASE_URL`` (ProxyAPI),
локальных моделей (sentence-transformers) нет.
"""
from __future__ import annotations

import time
from typing import Optional

from haystack import Document, Pipeline
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from loguru import logger

import config

MESSAGE_TYPE = "user_message"
DOC_TYPE = "document_chunk"


def type_user_filter(doc_type: str, user_id: int) -> dict:
    return {
        "operator": "AND",
        "conditions": [
            {"field": "meta.type", "operator": "==", "value": doc_type},
            {"field": "meta.user_id", "operator": "==", "value": user_id},
        ],
    }


def build_document_store() -> PineconeDocumentStore:
    return PineconeDocumentStore(
        api_key=Secret.from_env_var("PINECONE_API_KEY"),
        index=config.PINECONE_INDEX_NAME,
        namespace=config.PINECONE_NAMESPACE,
        dimension=config.EMBEDDING_DIM,
        metric="cosine",
        spec={"serverless": {"region": config.PINECONE_REGION, "cloud": config.PINECONE_CLOUD}},
    )


def build_doc_embedder() -> OpenAIDocumentEmbedder:
    return OpenAIDocumentEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        api_base_url=config.OPENAI_BASE_URL,
        model=config.OPENAI_EMBEDDING_MODEL,
        meta_fields_to_embed=[],
        batch_size=32,
    )


def build_text_embedder() -> OpenAITextEmbedder:
    return OpenAITextEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        api_base_url=config.OPENAI_BASE_URL,
        model=config.OPENAI_EMBEDDING_MODEL,
    )


class ContextStore:
    """Запись/удаление сообщений пользователя и управление документами в Pinecone."""

    def __init__(self, document_store=None, doc_embedder=None):
        self.document_store = document_store or build_document_store()
        self.doc_embedder = doc_embedder or build_doc_embedder()
        # Document -> embedding -> Pinecone
        self._indexing = Pipeline()
        self._indexing.add_component("embedder", self.doc_embedder)
        self._indexing.add_component(
            "writer", DocumentWriter(self.document_store, policy=DuplicatePolicy.OVERWRITE)
        )
        self._indexing.connect("embedder.documents", "writer.documents")

    def save_user_message(self, user_id: int, text: str, username: Optional[str] = None) -> None:
        """Сохраняет текст сообщения пользователя (ответы бота в память не пишутся)."""
        text = (text or "").strip()
        if not text:
            return
        meta = {"type": MESSAGE_TYPE, "user_id": user_id, "timestamp": int(time.time())}
        if username:
            meta["username"] = username
        logger.info("[pinecone] save user={} text={!r}", user_id, text[:80])
        self._indexing.run({"embedder": {"documents": [Document(content=text, meta=meta)]}})

    def list_files(self, user_id: int) -> dict[str, int]:
        """{имя файла: число чанков} для загруженных пользователем документов."""
        docs = self.document_store.filter_documents(filters=type_user_filter(DOC_TYPE, user_id))
        files: dict[str, int] = {}
        for d in docs:
            name = d.meta.get("file_name", "?")
            files[name] = files.get(name, 0) + 1
        return files

    def clear_messages(self, user_id: int) -> None:
        self.document_store.delete_by_filter(filters=type_user_filter(MESSAGE_TYPE, user_id))

    def clear_documents(self, user_id: int) -> None:
        self.document_store.delete_by_filter(filters=type_user_filter(DOC_TYPE, user_id))

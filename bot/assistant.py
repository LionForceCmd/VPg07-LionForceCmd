"""Логика ассистента без привязки к Telegram (удобно тестировать)."""
from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Optional

from haystack.dataclasses import ChatMessage
from loguru import logger

import config
from components import tools
from components.context import ContextStore
from pipelines.generation_pipeline import run_generation
from pipelines.ingestion_pipeline import run_ingestion


class Assistant:
    def __init__(self, store: ContextStore, generation_pipeline, ingestion_pipeline):
        self.store = store
        self.generation = generation_pipeline
        self.ingestion = ingestion_pipeline
        self._gen_lock = threading.Lock()     # один запрос агента за раз (общий слот outbox)
        self._ingest_lock = threading.Lock()  # Docling тяжёлый -- файлы обрабатываем по одному
        self.history: dict[int, deque] = defaultdict(lambda: deque(maxlen=config.SHORT_TERM_MESSAGES))

    # ------------------------------------------------------------ текст
    def handle_text(self, user_id: int, chat_id: int, text: str,
                    username: Optional[str] = None) -> tuple[str, Optional[str]]:
        """Возвращает (ответ, url картинки | None)."""
        # 1. generation_pipeline: поиск по документам и памяти -> агент.
        #    Идёт ДО записи сообщения, чтобы поиск не нашёл само это сообщение.
        with self._gen_lock:
            tools.outbox.set(None)
            try:
                answer, _docs = run_generation(
                    self.generation, user_id, text, list(self.history[chat_id]),
                    docs_top_k=config.DOCS_TOP_K, mem_top_k=config.CONTEXT_TOP_K,
                )
                image = tools.outbox.get()
            finally:
                tools.outbox.set(None)

        # 2. В Pinecone пишем только сообщение пользователя
        try:
            self.store.save_user_message(user_id, text, username)
        except Exception:
            logger.exception("Ошибка записи в Pinecone")

        # 3. Краткосрочная память (только в RAM)
        self.history[chat_id].append(ChatMessage.from_user(text))
        self.history[chat_id].append(ChatMessage.from_assistant(answer))
        return answer, image

    # ----------------------------------------------------------- файлы
    def handle_file(self, user_id: int, path: str, file_name: str) -> tuple[int, str]:
        """ingestion_pipeline: Docling -> чанки -> Pinecone + резюме. Возвращает (чанков, резюме)."""
        with self._ingest_lock:
            return run_ingestion(self.ingestion, path, file_name, user_id)

    # ------------------------------------------------------- управление
    def clear(self, user_id: int, chat_id: int) -> None:
        self.history.pop(chat_id, None)
        self.store.clear_messages(user_id)

    def clear_documents(self, user_id: int) -> None:
        self.store.clear_documents(user_id)

    def list_files(self, user_id: int) -> dict[str, int]:
        return self.store.list_files(user_id)

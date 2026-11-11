"""Точка входа: собирает компоненты и пайплайны и запускает Telegram-бота.

Запуск (из папки hay_v2_bot):  python main.py
"""
from __future__ import annotations

import sys

from loguru import logger

import config
from bot.assistant import Assistant
from bot.telegram_bot import create_bot
from components.context import ContextStore
from pipelines.generation_pipeline import build_generation_pipeline
from pipelines.ingestion_pipeline import build_ingestion_pipeline


def main() -> None:
    missing = [k for k in ("TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY", "PINECONE_API_KEY")
               if not getattr(config, k)]
    if missing:
        sys.exit(f"В .env не заданы: {', '.join(missing)}")

    store = ContextStore()  # Pinecone + эмбеддер документов
    generation = build_generation_pipeline(store.document_store)
    ingestion = build_ingestion_pipeline(store.document_store, store.doc_embedder)
    ingestion.warm_up()  # загрузит модели Docling заранее, а не на первом файле

    bot = create_bot(Assistant(store, generation, ingestion))
    logger.info("Бот запущен, индекс Pinecone: {}", config.PINECONE_INDEX_NAME)
    bot.infinity_polling()


if __name__ == "__main__":
    main()

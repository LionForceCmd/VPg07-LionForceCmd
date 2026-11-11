"""Компонент: собирает сообщения для агента из найденных документов, памяти и истории."""
from __future__ import annotations

from typing import Optional

from haystack import Document, component
from haystack.dataclasses import ChatMessage


def format_source(d: Document) -> str:
    m = d.meta or {}
    where = m.get("file_name", "?")
    if m.get("pages"):
        where += f", стр. {m['pages']}"
    elif m.get("chunk_index") is not None:
        where += f", чанк {int(m['chunk_index']) + 1}"
    return where


@component
class ContextBuilder:
    @component.output_types(messages=list[ChatMessage])
    def run(
        self,
        query: str,
        documents: list[Document],
        memories: list[Document],
        history: Optional[list[ChatMessage]] = None,
    ) -> dict:
        mem_lines = [f"- {m.content}" for m in memories]
        mem_ctx = "Контекст из памяти (прошлые сообщения пользователя):\n" + (
            "\n".join(mem_lines) if mem_lines else "(пусто)"
        )
        doc_lines = [f"[{i}] ({format_source(d)})\n{d.content}" for i, d in enumerate(documents, 1)]
        doc_ctx = "Фрагменты из загруженных пользователем документов:\n" + (
            "\n\n".join(doc_lines) if doc_lines else "(документов нет или ничего подходящего не найдено)"
        )
        messages = [
            ChatMessage.from_system(mem_ctx),
            ChatMessage.from_system(doc_ctx),
            *(history or []),
            ChatMessage.from_user(query),
        ]
        return {"messages": messages}

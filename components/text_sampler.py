"""Компонент: из списка чанков собирает один текст для резюме (с лимитом по символам)."""
from __future__ import annotations

from haystack import Document, component


@component
class TextSampler:
    def __init__(self, max_chars: int = 12000):
        self.max_chars = max_chars

    @component.output_types(text=str, file_name=str)
    def run(self, documents: list[Document]) -> dict:
        parts: list[str] = []
        total = 0
        for d in documents:
            piece = (d.content or "").strip()
            if not piece:
                continue
            if total + len(piece) > self.max_chars:
                piece = piece[: max(0, self.max_chars - total)]
            parts.append(piece)
            total += len(piece)
            if total >= self.max_chars:
                break
        file_name = documents[0].meta.get("file_name", "") if documents else ""
        return {"text": "\n\n".join(parts), "file_name": file_name}

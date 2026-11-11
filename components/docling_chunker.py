"""Компонент Haystack: файл -> Docling -> структурные чанки (Document).

Docling разбирает PDF/DOCX/PPTX/XLSX/HTML/MD/... (с OCR для сканов), строит
``DoclingDocument`` и режет его ``HybridChunker``-ом по структуре документа
(заголовки, абзацы, таблицы) с ограничением по токенам.

Метаданные каждого чанка (плоские -- Pinecone не принимает вложенные объекты):
``file_name``, ``chunk_index``, ``chunk_total``, ``page`` (первая страница, если
известна), ``pages`` (строка "1,2"), ``headings`` (строка), ``type="document_chunk"``.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from haystack import Document, component
from loguru import logger

import config

DOC_TYPE = "document_chunk"


def _page_numbers(chunk: Any) -> list[int]:
    pages: set[int] = set()
    for item in getattr(chunk.meta, "doc_items", None) or []:
        for prov in getattr(item, "prov", None) or []:
            if getattr(prov, "page_no", None) is not None:
                pages.add(int(prov.page_no))
    return sorted(pages)


def _build_chunker(max_tokens: int):
    """HybridChunker с токенизатором tiktoken (без скачивания HF-моделей)."""
    from docling.chunking import HybridChunker

    try:
        import tiktoken
        from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

        tokenizer = OpenAITokenizer(
            tokenizer=tiktoken.get_encoding("cl100k_base"), max_tokens=max_tokens
        )
        return HybridChunker(tokenizer=tokenizer)
    except Exception as e:  # старые версии docling-core / нет tiktoken
        logger.warning("OpenAITokenizer недоступен ({}), использую HybridChunker по умолчанию", e)
        return HybridChunker(max_tokens=max_tokens)


@component
class DoclingChunker:
    """Вход: ``sources`` (пути к файлам) -> выход: ``documents`` (чанки с метаданными)."""

    def __init__(self, max_tokens: int = 512, do_ocr: bool = True, extra_meta: Optional[dict] = None):
        self.max_tokens = max_tokens
        self.do_ocr = do_ocr
        self.extra_meta = extra_meta or {}
        self._converter = None
        self._chunker = None

    # Тяжёлые объекты (модели layout/OCR) создаём лениво и один раз
    def warm_up(self) -> None:
        if self._converter is not None:
            return
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pdf_opts = PdfPipelineOptions()
        pdf_opts.do_ocr = self.do_ocr
        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
        )
        self._chunker = _build_chunker(self.max_tokens)

    @component.output_types(documents=list[Document])
    def run(self, sources: list[str], file_names: Optional[list[str]] = None,
            user_id: Optional[int] = None) -> dict:
        self.warm_up()
        docs: list[Document] = []
        for i, src in enumerate(sources):
            path = Path(src)
            name = (file_names[i] if file_names and i < len(file_names) else path.name)
            docs.extend(self._process_one(path, name, user_id))
        return {"documents": docs}

    # ---------------------------------------------------------------- внутреннее
    def _process_one(self, path: Path, file_name: str, user_id: Optional[int]) -> list[Document]:
        tmp_dir = None
        try:
            if path.suffix.lower() == ".txt":  # Docling не знает .txt -- читаем как Markdown
                tmp_dir = tempfile.mkdtemp(prefix="docling_")
                md = Path(tmp_dir) / (path.stem + ".md")
                shutil.copyfile(path, md)
                path = md
            logger.info("[docling] convert {}", file_name)
            dl_doc = self._converter.convert(str(path)).document

            chunks = list(self._chunker.chunk(dl_doc=dl_doc))
            result: list[Document] = []
            for chunk in chunks:
                # contextualize() добавляет к тексту заголовки -- лучше для поиска и эмбеддингов
                text = self._chunker.contextualize(chunk=chunk).strip()
                if not text:
                    continue
                pages = _page_numbers(chunk)
                meta: dict[str, Any] = {
                    "type": DOC_TYPE,
                    "file_name": file_name,
                    "headings": " > ".join(getattr(chunk.meta, "headings", None) or [])[:500],
                    **self.extra_meta,
                }
                if user_id is not None:
                    meta["user_id"] = user_id
                if pages:
                    meta["page"] = pages[0]
                    meta["pages"] = ",".join(map(str, pages))
                result.append(Document(content=text, meta=meta))
            for idx, d in enumerate(result):
                d.meta["chunk_index"] = idx
                d.meta["chunk_total"] = len(result)
            logger.info("[docling] {} -> {} чанков", file_name, len(result))
            return result
        finally:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

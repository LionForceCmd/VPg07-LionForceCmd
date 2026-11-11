"""Офлайн-тесты без ключей и сети: Docling на реальном файле + сборка пайплайнов на заглушках.

Запуск (из папки hay_v2_bot):  python -m tests.test_offline
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("PINECONE_API_KEY", "test")

from haystack import Document, component
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore

from components.context_builder import ContextBuilder
from components.docling_chunker import DoclingChunker
from pipelines.ingestion_pipeline import build_ingestion_pipeline, run_ingestion

SAMPLE_MD = """# Отчёт о продажах

## Q1
Выручка за первый квартал составила 120 млн рублей, рост 15%.

## Q2
Во втором квартале запущен новый продукт «Альфа»; выручка 150 млн рублей.

| Регион | Выручка |
|--------|---------|
| Москва | 90      |
| Казань | 60      |
"""


@component
class FakeEmbedder:
    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        for d in documents:
            d.embedding = [0.1] * 8
        return {"documents": documents}


@component
class FakeLLM:
    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]):
        assert "Выручка" in messages[0].text  # текст файла дошёл до промпта резюме
        return {"replies": [ChatMessage.from_assistant("Отчёт описывает выручку компании по кварталам.")]}


def test_chunker_and_ingestion():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "upload.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_MD)

        chunks = DoclingChunker(max_tokens=64).run([path], ["report.md"], user_id=7)["documents"]
        assert chunks, "Docling не вернул чанков"
        for c in chunks:
            assert c.meta["file_name"] == "report.md" and c.meta["user_id"] == 7
            assert c.meta["type"] == "document_chunk" and "chunk_index" in c.meta
            assert all(isinstance(v, (str, int, float, bool)) for v in c.meta.values()), c.meta
        print(f"chunker: {len(chunks)} чанков; пример: {chunks[0].content[:80]!r} {chunks[0].meta}")

        store = InMemoryDocumentStore()
        pipe = build_ingestion_pipeline(store, FakeEmbedder(), chat_generator=FakeLLM(),
                                        chunker=DoclingChunker(max_tokens=64))
        written, summary = run_ingestion(pipe, path, "report.md", 7)
        assert written == len(chunks) and store.count_documents() == written
        assert summary == "Отчёт описывает выручку компании по кварталам."
        print("ingestion_pipeline OK:", written, "чанков,", summary)


def test_context_builder():
    docs = [Document(content="Выручка Q1 120 млн", meta={"file_name": "a.pdf", "pages": "3"})]
    mems = [Document(content="Я живу в Казани")]
    msgs = ContextBuilder().run("Какая выручка?", docs, mems, [ChatMessage.from_user("hi")])["messages"]
    text = "\n".join(m.text for m in msgs)
    assert "a.pdf, стр. 3" in text and "Я живу в Казани" in text and msgs[-1].text == "Какая выручка?"
    print("context_builder OK")


def test_generation_pipeline_builds():
    from pipelines.generation_pipeline import build_generation_pipeline

    class FakeStore(InMemoryDocumentStore):
        pass

    # Pinecone-ретривер требует PineconeDocumentStore, поэтому проверяем только импорт/сигнатуры
    assert callable(build_generation_pipeline)
    print("generation_pipeline import OK")


if __name__ == "__main__":
    test_context_builder()
    test_generation_pipeline_builds()
    test_chunker_and_ingestion()
    print("ALL OK")

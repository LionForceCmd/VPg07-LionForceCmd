# Архитектура hay_v2_bot

Ядро — два Haystack-пайплайна, собранных из компонентов. Telegram-слой тонкий и
ничего не знает о Docling/Pinecone.

## ingestion_pipeline (файл → знания)

```
              ┌──documents──> embedder ──> writer  ──> Pinecone
 chunker ─────┤
 (Docling)    └──documents──> sampler ──> prompt ──> llm ──> резюме (1 предложение)
```

| Компонент | Что делает |
|---|---|
| `DoclingChunker` | `DocumentConverter` (PDF/DOCX/PPTX/XLSX/HTML/MD/картинки, OCR) → `DoclingDocument` → `HybridChunker` (по структуре, лимит токенов, tiktoken). К тексту чанка добавляются заголовки. |
| `OpenAIDocumentEmbedder` | эмбеддинги через ProxyAPI (`OPENAI_BASE_URL`), `text-embedding-3-small` |
| `DocumentWriter` | запись в Pinecone |
| `TextSampler` | склеивает начало документа (до `SUMMARY_MAX_CHARS`) для резюме |
| `ChatPromptBuilder` + `OpenAIChatGenerator` | резюме ровно в одно предложение |

Метаданные чанка в Pinecone: `type=document_chunk`, `user_id`, `file_name`,
`chunk_index`, `chunk_total`, `page`/`pages` (если известны), `headings`.

## generation_pipeline (вопрос → ответ)

```
text_embedder ─> doc_retriever ─documents─┐
              └> mem_retriever ─memories──┴> context_builder ─messages─> agent ─> ответ
```

* `doc_retriever` ищет чанки документов пользователя (`type=document_chunk`, `user_id`);
* `mem_retriever` ищет его прошлые сообщения (`type=user_message`);
* `ContextBuilder` собирает системные сообщения («память», «фрагменты документов»), историю и вопрос;
* `Agent` (с инструментами кошки/собаки/погода) отвечает, ссылаясь на источники.

Поиск выполняется до записи нового сообщения, чтобы не находить его самого.
В Pinecone попадают только сообщения пользователя и чанки файлов — не ответы бота.

## Слои

`bot/telegram_bot.py` (события Telegram) → `bot/assistant.py` (логика, блокировки,
краткосрочная память) → `pipelines/*` (оркестрация) → `components/*` (отдельные шаги).
Пайплайны принимают хранилище/эмбеддеры/генераторы параметрами — поэтому в тестах
подставляются заглушки.

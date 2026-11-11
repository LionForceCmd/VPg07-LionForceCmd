"""Настройки из .env. Всё обращение к OpenAI идёт через OPENAI_BASE_URL (ProxyAPI)."""
import os

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# --- LLM / эмбеддинги: только через base URL прокси, локальных эмбеддингов нет
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

# --- Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "haystack-assistant-v2")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "assistant")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# --- Поиск контекста
CONTEXT_TOP_K = int(os.getenv("CONTEXT_TOP_K", "5"))          # прошлые сообщения
DOCS_TOP_K = int(os.getenv("DOCS_TOP_K", "5"))                # чанки документов
SHORT_TERM_MESSAGES = int(os.getenv("SHORT_TERM_MESSAGES", "10"))

# --- Docling
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "512"))   # размер чанка (токены)
DOCLING_DO_OCR = os.getenv("DOCLING_DO_OCR", "true").lower() in ("1", "true", "yes")
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "20"))              # лимит Bot API на скачивание = 20 МБ
SUMMARY_MAX_CHARS = int(os.getenv("SUMMARY_MAX_CHARS", "12000"))  # сколько текста отдаём на резюме

# Расширения, которые принимает Docling (+ .txt, который мы превращаем в .md)
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".md", ".txt", ".csv", ".adoc",
    ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp",
}

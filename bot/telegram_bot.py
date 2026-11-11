"""Telegram-слой: хендлеры команд, текста и файлов. Логика -- в bot.assistant."""
from __future__ import annotations

import os
import tempfile

import telebot
from loguru import logger

import config
from bot.assistant import Assistant

HELP_TEXT = (
    "Я персональный ассистент на Haystack + Docling + Pinecone. Умею:\n"
    "• читать твои файлы (PDF, DOCX, PPTX, XLSX, HTML, MD, TXT, CSV, картинки) — "
    "пришли файл, я его изучу и коротко перескажу 📄\n"
    "• отвечать на вопросы по загруженным документам 🔎\n"
    "• факт о кошках 🐱, анализ случайной собаки 🐶, погода в любом городе ☀️\n"
    "• помнить, что ты мне писал\n\n"
    "Команды: /start, /help, /files (мои документы), "
    "/clear (стереть память и историю), /cleardocs (удалить мои документы)."
)

FILE_RECEIVED = "Файл получен. Запускаю анализ и сохранение. Это может занять немного времени…"
FILE_DONE = "Готово. Я изучил этот файл, теперь можем его обсудить."


def create_bot(assistant: Assistant) -> telebot.TeleBot:
    bot = telebot.TeleBot(config.TELEGRAM_BOT_TOKEN)

    @bot.message_handler(commands=["start", "help"])
    def on_start(m):
        bot.reply_to(m, HELP_TEXT)

    @bot.message_handler(commands=["clear"])
    def on_clear(m):
        assistant.clear(m.from_user.id, m.chat.id)
        bot.reply_to(m, "Готово: история и память о тебе очищены (документы остались — /cleardocs).")

    @bot.message_handler(commands=["cleardocs"])
    def on_clear_docs(m):
        assistant.clear_documents(m.from_user.id)
        bot.reply_to(m, "Готово: все загруженные тобой документы удалены из базы.")

    @bot.message_handler(commands=["files"])
    def on_files(m):
        files = assistant.list_files(m.from_user.id)
        if not files:
            bot.reply_to(m, "Документов пока нет. Пришли файл — я его изучу.")
            return
        lines = [f"• {name} ({n} фрагм.)" for name, n in sorted(files.items())]
        bot.reply_to(m, "Твои документы:\n" + "\n".join(lines))

    @bot.message_handler(content_types=["document"])
    def on_document(m):
        doc = m.document
        name = doc.file_name or "document"
        ext = os.path.splitext(name)[1].lower()
        if ext not in config.SUPPORTED_EXTENSIONS:
            bot.reply_to(m, "Этот формат не поддерживается. Пришли: "
                            + ", ".join(sorted(e.lstrip(".") for e in config.SUPPORTED_EXTENSIONS)))
            return
        if doc.file_size and doc.file_size > config.MAX_FILE_MB * 1024 * 1024:
            bot.reply_to(m, f"Файл слишком большой (максимум {config.MAX_FILE_MB} МБ).")
            return

        bot.reply_to(m, FILE_RECEIVED)
        bot.send_chat_action(m.chat.id, "typing")
        try:
            info = bot.get_file(doc.file_id)
            data = bot.download_file(info.file_path)
            with tempfile.TemporaryDirectory(prefix="tg_file_") as tmp:
                path = os.path.join(tmp, "upload" + ext)  # безопасное имя на диске
                with open(path, "wb") as f:
                    f.write(data)
                chunks, summary = assistant.handle_file(m.from_user.id, path, name)
        except Exception:
            logger.exception("Ошибка обработки файла {}", name)
            bot.reply_to(m, "Не удалось обработать файл. Проверь, что он не повреждён, и попробуй ещё раз.")
            return

        if chunks == 0:
            bot.reply_to(m, "Не нашёл в файле текста (возможно, он пустой или это картинка без текста).")
            return
        bot.send_message(m.chat.id, FILE_DONE)
        bot.send_message(m.chat.id, summary or "(не удалось составить резюме)")

    @bot.message_handler(func=lambda m: bool(m.text) and not m.text.startswith("/"))
    def on_text(m):
        bot.send_chat_action(m.chat.id, "typing")
        try:
            answer, image = assistant.handle_text(m.from_user.id, m.chat.id, m.text, m.from_user.username)
        except Exception:
            logger.exception("Ошибка обработки сообщения")
            bot.reply_to(m, "Извини, что-то пошло не так. Попробуй ещё раз.")
            return
        if image:
            bot.send_photo(m.chat.id, image, caption=answer[:1024], reply_to_message_id=m.message_id)
            if len(answer) > 1024:
                bot.send_message(m.chat.id, answer[1024:])
        else:
            bot.reply_to(m, answer)

    return bot

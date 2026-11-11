"""LLM-компоненты: чат-генератор через прокси, агент с инструментами, промпт резюме."""
from __future__ import annotations

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

import config
from components import tools

SYSTEM_PROMPT = """Ты — дружелюбный персональный ассистент в Telegram. Отвечай кратко, \
на языке пользователя, как настоящий помощник: учитывай контекст разговора и то, \
что пользователь говорил раньше.

Тебе доступны инструменты — вызывай их, когда пользователь об этом просит:
- get_cat_fact — случайный факт о кошках («расскажи факт о кошках»);
- get_dog_image — просто показать случайную картинку собаки;
- analyze_dog_image — получить картинку собаки, определить породу и рассказать её \
историю (картинка отправляется пользователю автоматически);
- get_weather(city) — текущая погода в городе («какая погода в Москве?»). Если город \
не назван, возьми его из контекста прошлых сообщений, а если там его нет — спроси.

Правила:
- Не выдумывай погоду и факты — для них всегда вызывай инструмент.
- Раздел «Контекст из памяти» — это то, что пользователь писал раньше. Опирайся на него, \
когда вопрос касается личных данных или прошлых разговоров.
- Раздел «Фрагменты из загруженных пользователем документов» — выдержки из его файлов. \
Если вопрос касается содержимого документов, отвечай строго по этим фрагментам и \
называй источник (имя файла, страницу). Если в фрагментах нет ответа, честно скажи, \
что в документах этого не нашлось, и не выдумывай.
- Если ни в памяти, ни в документах нет ответа, честно скажи, что не знаешь."""

SUMMARY_PROMPT = """Ниже — содержимое файла «{{ file_name }}». Напиши его краткое резюме \
РОВНО ОДНИМ предложением на русском языке (даже если документ на другом языке). \
Без вступлений, списков и markdown — только это одно предложение.

Содержимое:
{{ text }}"""


def build_chat_generator() -> OpenAIChatGenerator:
    return OpenAIChatGenerator(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        api_base_url=config.OPENAI_BASE_URL,  # ProxyAPI
        model=config.OPENAI_CHAT_MODEL,
    )


def build_agent() -> Agent:
    return Agent(
        chat_generator=build_chat_generator(),
        tools=tools.build_tools(),
        system_prompt=SYSTEM_PROMPT,
        exit_conditions=["text"],
        max_agent_steps=8,
    )


def summary_template() -> list[ChatMessage]:
    return [ChatMessage.from_user(SUMMARY_PROMPT)]

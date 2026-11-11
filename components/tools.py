"""Инструменты (tools) Haystack-агента.

* get_cat_fact          -- случайный факт о кошках (catfact.ninja)
* get_dog_image         -- случайная картинка собаки (dog.ceo)
* analyze_dog_image     -- картинка собаки + описание породы через OpenAI vision
* get_weather           -- НОВЫЙ инструмент: текущая погода в городе (Open-Meteo, без ключа)

Картинку, которую нашёл инструмент, он кладёт в "почтовый ящик" запроса
(``outbox``), а бот отправляет её в Telegram вместе с ответом как caption.
"""
from __future__ import annotations

import base64
import threading
from typing import Optional

import requests
from haystack.tools import Tool
from loguru import logger
from openai import OpenAI

import config

HTTP_TIMEOUT = 15

class _Outbox:
    """Картинка, подготовленная инструментами для текущего запроса.

    Haystack вызывает инструменты в пуле потоков, поэтому ContextVar не подходит --
    используем общий слот. Бот обрабатывает запросы под замком (см. bot.Assistant).
    """

    def __init__(self) -> None:
        self._url: Optional[str] = None
        self._lock = threading.Lock()

    def set(self, url: Optional[str]) -> None:
        with self._lock:
            self._url = url

    def get(self) -> Optional[str]:
        with self._lock:
            return self._url


outbox = _Outbox()

# Коды погоды WMO -> описание
WMO_CODES = {
    0: "ясно", 1: "в основном ясно", 2: "переменная облачность", 3: "пасмурно",
    45: "туман", 48: "изморозь", 51: "лёгкая морось", 53: "морось", 55: "сильная морось",
    61: "небольшой дождь", 63: "дождь", 65: "сильный дождь", 66: "ледяной дождь",
    67: "сильный ледяной дождь", 71: "небольшой снег", 73: "снег", 75: "сильный снег",
    77: "снежные зёрна", 80: "небольшие ливни", 81: "ливни", 82: "сильные ливни",
    85: "снегопад", 86: "сильный снегопад", 95: "гроза", 96: "гроза с градом",
    99: "сильная гроза с градом",
}


def _openai_client() -> OpenAI:
    return OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)


# ---------------------------------------------------------------- cat fact
def get_cat_fact() -> str:
    logger.info("[tool] get_cat_fact")
    try:
        r = requests.get("https://catfact.ninja/fact", timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        fact = r.json()["fact"]
        logger.info("[tool] get_cat_fact -> {}", fact)
        return fact
    except Exception as e:
        logger.exception("get_cat_fact failed")
        return f"Ошибка получения факта о кошках: {e}"


# --------------------------------------------------------------- dog image
def _fetch_dog_image_url() -> str:
    r = requests.get("https://dog.ceo/api/breeds/image/random", timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "success":
        raise RuntimeError(f"dog.ceo: {data}")
    return data["message"]


def get_dog_image() -> str:
    logger.info("[tool] get_dog_image")
    try:
        url = _fetch_dog_image_url()
        outbox.set(url)
        logger.info("[tool] get_dog_image -> {}", url)
        return f"Случайная картинка собаки получена: {url}. Она будет отправлена пользователю."
    except Exception as e:
        logger.exception("get_dog_image failed")
        return f"Ошибка получения картинки собаки: {e}"


def analyze_dog_image() -> str:
    """Берёт картинку собаки из API, отправляет в OpenAI и возвращает описание породы."""
    logger.info("[tool] analyze_dog_image")
    try:
        url = _fetch_dog_image_url()
        img = requests.get(url, timeout=HTTP_TIMEOUT)
        img.raise_for_status()
        mime = img.headers.get("Content-Type", "image/jpeg").split(";")[0]
        data_url = f"data:{mime};base64,{base64.b64encode(img.content).decode()}"
        resp = _openai_client().chat.completions.create(
            model=config.OPENAI_CHAT_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Определи породу собаки на фото и кратко (3-5 предложений) "
                        "расскажи историю происхождения этой породы. Отвечай по-русски.")},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
        )
        description = resp.choices[0].message.content.strip()
        outbox.set(url)
        logger.info("[tool] analyze_dog_image -> {} | {}", url, description[:120])
        return description
    except Exception as e:
        logger.exception("analyze_dog_image failed")
        return f"Ошибка анализа картинки собаки: {e}"


# ----------------------------------------------------------------- weather
def get_weather(city: str) -> str:
    """Текущая погода в городе через бесплатный Open-Meteo (геокодинг + прогноз)."""
    logger.info("[tool] get_weather(city={!r})", city)
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "ru"},
            timeout=HTTP_TIMEOUT,
        )
        geo.raise_for_status()
        results = geo.json().get("results")
        if not results:
            return f"Город «{city}» не найден."
        place = results[0]
        w = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                           "wind_speed_10m,weather_code",
                "wind_speed_unit": "ms",
                "timezone": "auto",
            },
            timeout=HTTP_TIMEOUT,
        )
        w.raise_for_status()
        cur = w.json()["current"]
        desc = WMO_CODES.get(cur["weather_code"], f"код погоды {cur['weather_code']}")
        result = (
            f"Погода в {place['name']} ({place.get('country', '')}): {desc}, "
            f"{cur['temperature_2m']}°C (ощущается как {cur['apparent_temperature']}°C), "
            f"влажность {cur['relative_humidity_2m']}%, ветер {cur['wind_speed_10m']} м/с."
        )
        logger.info("[tool] get_weather -> {}", result)
        return result
    except Exception as e:
        logger.exception("get_weather failed")
        return f"Ошибка получения погоды: {e}"


# ------------------------------------------------------- Haystack Tool list
def build_tools() -> list[Tool]:
    no_params = {"type": "object", "properties": {}}
    return [
        Tool(
            name="get_cat_fact",
            description="Возвращает случайный интересный факт о кошках. "
                        "Используй, когда пользователь просит факт о кошках.",
            parameters=no_params,
            function=get_cat_fact,
        ),
        Tool(
            name="get_dog_image",
            description="Получает случайную картинку собаки и отправляет её пользователю "
                        "без анализа. Используй, если просят просто показать собаку.",
            parameters=no_params,
            function=get_dog_image,
        ),
        Tool(
            name="analyze_dog_image",
            description="Получает случайную картинку собаки, анализирует её через "
                        "OpenAI, определяет породу и рассказывает её историю. Картинка "
                        "автоматически отправляется пользователю. Используй, когда "
                        "просят проанализировать/определить породу собаки по картинке.",
            parameters=no_params,
            function=analyze_dog_image,
        ),
        Tool(
            name="get_weather",
            description="Возвращает текущую погоду (температура, ощущаемая температура, "
                        "влажность, ветер, осадки) в указанном городе. Используй, когда "
                        "пользователь спрашивает о погоде.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Название города, например 'Москва' или 'London'",
                    }
                },
                "required": ["city"],
            },
            function=get_weather,
        ),
    ]

"""CSV-backed multilingual UI strings for the Kisan Telegram Bot."""

import csv
import os
from functools import lru_cache

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "हिंदी",
    "od": "ଓଡ଼ିଆ",
    "bn": "বাংলা",
    "mr": "मराठी",
    "te": "తెలుగు",
    "ta": "தமிழ்",
}


def _csv_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        "translations.csv",
    )


@lru_cache(maxsize=1)
def load_translations() -> dict:
    translations = {}
    path = _csv_path()
    if not os.path.exists(path):
        return translations

    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            language = (row.get("language_code") or "").strip()
            key = (row.get("key") or "").strip()
            text = row.get("text") or ""
            if language and key:
                translations[(language, key)] = text
    return translations


def get_text(key: str, language: str = "en") -> str:
    language = language if language in SUPPORTED_LANGUAGES else "en"
    translations = load_translations()
    return translations.get((language, key), translations.get(("en", key), key))

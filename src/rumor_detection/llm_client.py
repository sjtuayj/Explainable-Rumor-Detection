"""
OpenAI-compatible client for the SJTU model API.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False


load_dotenv()

SJTU_BASE_URL = os.getenv("SJTU_BASE_URL", "https://models.sjtu.edu.cn/api/v1")
DEFAULT_LLM_MODEL = os.getenv("SJTU_MODEL", "deepseek-chat")


def get_llm_client() -> OpenAI:
    api_key = os.getenv("SJTU_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SJTU_API_KEY is not set. Add it to a .env file or export it before "
            "calling the LLM API."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai is not installed. Run `pip install -r requirements.txt`.") from exc
    return OpenAI(api_key=api_key, base_url=SJTU_BASE_URL)

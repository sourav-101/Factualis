from __future__ import annotations

import os

from openai import OpenAI

from .answer_format import response_format

SGLANG_URL = os.environ.get("SGLANG_URL", "http://localhost:30000/v1")


def get_client() -> OpenAI:
    return OpenAI(base_url=SGLANG_URL, api_key="EMPTY")


def chat(messages: list[dict], model: str = "default", constrained: bool = True, max_tokens: int = 1536, client: OpenAI | None = None) -> str:
    client = client or get_client()
    kwargs = {"response_format": response_format()} if constrained else {}
    r = client.chat.completions.create(model=model, messages=messages, temperature=0.0, max_tokens=max_tokens, **kwargs)
    return r.choices[0].message.content or ""

"""Generic chat-completion client for the inference pipeline's "LLM ->
grounded answer" box. Deliberately separate from
baselines/_openrouter_client.py: that module is hardcoded to one fixed
TTL-prediction prompt for the Week-3/4 benchmark baselines; this one takes
arbitrary chat messages, since the inference pipeline needs to send a
grounded-QA prompt (or, in principle, other prompts), not a TTL guess. Same
OpenRouter routing (one key, one endpoint, model by slug) for consistency
with the rest of the project.
"""
import os

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o"


def chat_completion(messages: list[dict], model: str = DEFAULT_MODEL, api_key: str | None = None,
                     temperature: float = 0.0, max_tokens: int = 300) -> tuple[str, dict]:
    """messages: standard OpenAI-style [{"role": ..., "content": ...}, ...].
    Returns (response_text, usage_dict). Raises SystemExit if no API key is
    available (env var or explicit arg), matching the baselines' convention
    of failing loudly and immediately rather than silently no-op-ing."""
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set in environment -- export it and re-run.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/frustratedengineer-gif/pgp",
        "X-Title": "MemoryLifeBench",
    }
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    text = body["choices"][0]["message"]["content"].strip()
    usage = body.get("usage") or {}
    return text, usage

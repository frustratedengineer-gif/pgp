"""
Baseline: LLM-prompted TTL using GPT-4o (via OpenRouter) -- the "current LLM"
a reviewer actually means when they ask "why not just ask ChatGPT?", as
opposed to the local Qwen2.5-7B in llm_prompted_ttl.py.

Requires OPENROUTER_API_KEY in the environment. Never read from a committed
file -- export it in your shell before running scripts/run_baseline.py.
"""
from baselines._openrouter_client import run_via_openrouter

DEFAULT_MODEL = "openai/gpt-4o"


def run(args) -> None:
    run_via_openrouter(args, "chatgpt_prompted_ttl", DEFAULT_MODEL, "chatgpt_model")

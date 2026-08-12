"""
Baseline: LLM-prompted TTL using Gemini (via OpenRouter) -- the other half
of "current LLM" (ChatGPT + Gemini) a reviewer means, as opposed to the
local Qwen2.5-7B in llm_prompted_ttl.py.

Requires OPENROUTER_API_KEY in the environment. Never read from a committed
file -- export it in your shell before running scripts/run_baseline.py.

Default model is a placeholder ("google/gemini-2.5-pro") -- if OpenRouter
has retired that slug, check https://openrouter.ai/models for the current
Gemini Pro identifier and pass --gemini-model to override.
"""
from baselines._openrouter_client import run_via_openrouter

DEFAULT_MODEL = "google/gemini-2.5-pro"


def run(args) -> None:
    run_via_openrouter(args, "gemini_prompted_ttl", DEFAULT_MODEL, "gemini_model")

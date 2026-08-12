"""
Shared client for the two real-frontier-LLM baselines (chatgpt_prompted_ttl.py,
gemini_prompted_ttl.py). Both route through OpenRouter (https://openrouter.ai)
-- one API key, one OpenAI-compatible endpoint, model selected by name (e.g.
"openai/gpt-4o", "google/gemini-2.5-pro"). This is what lets us compare against
both "current LLM" families without juggling two different native SDKs/auth
schemes.

Also accumulates per-call token usage (OpenRouter returns OpenAI-style
`usage: {prompt_tokens, completion_tokens, total_tokens}`) so token spend can
be reported as its own performance metric -- see
scripts/evaluate.py:collect_token_usage and results/tables/llm_token_usage.md.

Responses are cached on disk keyed by (model, prompt) -- Week-4 experiments
(multi-seed reruns, consistency checks across paraphrases) re-query the same
records repeatedly, and re-billing OpenRouter for an identical prompt every
time would be both wasteful and non-deterministic (real APIs aren't bit-exact
even at temperature=0). Cache hits don't count toward this run's token spend.
"""
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from memorylife.utils.io import load_jsonl

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
FALLBACK_TTL = 7.0
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

# identical prompt to the local-model baseline on purpose -- isolates "which
# model" as the only variable across all LLM-prompted-TTL comparisons
PROMPT_TEMPLATE = """You are estimating how long a piece of memory stays useful for a personal AI assistant.

Memory statement: "{text}"

Question: starting from when this was said, how many days from now is this fact likely to remain true and useful to remember, before it becomes outdated or irrelevant? Some facts are permanent (use a large number like 3650 for "essentially forever"), some last weeks or months, and some are only relevant for a day or two.

Answer with ONLY a single integer number of days. No words, no explanation."""


def _cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}||{prompt}".encode()).hexdigest()


def _query_one(api_key, model, memory_id, text, cache_dir: Path | None = None,
                prompt_template: str = PROMPT_TEMPLATE):
    prompt = prompt_template.format(text=text[:500])

    cache_path = None
    if cache_dir is not None:
        cache_path = cache_dir / f"{_cache_key(model, prompt)}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            return memory_id, cached["val"], {**cached.get("usage", {}), "cached": True}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # optional but polite: identifies the app to OpenRouter, no effect on billing
        "HTTP-Referer": "https://github.com/frustratedengineer-gif/pgp",
        "X-Title": "MemoryLifeBench",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 16,
    }
    for attempt in range(4):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            body = resp.json()
            raw = body["choices"][0]["message"]["content"].strip()
            usage = body.get("usage") or {}
            m = NUMBER_RE.search(raw)
            val = max(float(m.group()), 0.01) if m else FALLBACK_TTL
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps({"val": val, "usage": usage, "raw": raw}))
            return memory_id, val, usage
        except Exception:
            if attempt == 3:
                return memory_id, FALLBACK_TTL, {}
            time.sleep(2 ** attempt)


def run_via_openrouter(args, method_name: str, default_model: str, model_arg_name: str,
                        cache_dir: Path | None = None, prompt_template: str = PROMPT_TEMPLATE,
                        score_suffix: str = "") -> None:
    """cache_dir defaults to artifacts/llm_cache/<method_name>/. score_suffix
    lets a caller (e.g. the consistency check, which re-queries the same
    records with paraphrased prompts) write to <method_name><suffix>_<split>.json
    instead of overwriting the main baseline's score file."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set in environment -- export it and re-run.")

    model = getattr(args, model_arg_name, None) or default_model
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is None:
        cache_dir = Path("artifacts/llm_cache") / method_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    usage_report = {}

    for split_name in args.splits:
        records = load_jsonl(Path(args.data_dir) / f"{split_name}_survival.jsonl")
        scores = {}
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0, "cache_hits": 0}
        lock = threading.Lock()

        t0 = time.time()
        # lower concurrency than the local-model baseline -- respect the
        # real API's rate limits instead of hammering it
        with ThreadPoolExecutor(max_workers=min(args.llm_workers, 4)) as ex:
            futures = {
                ex.submit(_query_one, api_key, model, r["memory_id"], r["text"], cache_dir, prompt_template): r["memory_id"]
                for r in records
            }
            for i, fut in enumerate(as_completed(futures), 1):
                mid, val, usage = fut.result()
                scores[mid] = val
                with lock:
                    totals["calls"] += 1
                    if usage.get("cached"):
                        totals["cache_hits"] += 1
                    else:
                        totals["prompt_tokens"] += usage.get("prompt_tokens", 0)
                        totals["completion_tokens"] += usage.get("completion_tokens", 0)
                        totals["total_tokens"] += usage.get("total_tokens", 0)
                if i % 200 == 0:
                    print(f"  {split_name}: {i}/{len(records)} ({time.time()-t0:.0f}s)")

        elapsed = time.time() - t0
        with open(out_dir / f"{method_name}{score_suffix}_{split_name}.json", "w") as f:
            json.dump(scores, f)

        new_calls = totals["calls"] - totals["cache_hits"]
        avg_tok = totals["total_tokens"] / new_calls if new_calls else 0
        usage_report[split_name] = {
            **totals,
            "avg_tokens_per_call": round(avg_tok, 1),
            "elapsed_seconds": round(elapsed, 1),
        }
        print(f"{split_name}: wrote {len(scores)} scores in {elapsed:.0f}s -> "
              f"{method_name}{score_suffix}_{split_name}.json (model={model}, "
              f"tokens={totals['total_tokens']} over {new_calls} new calls, "
              f"{totals['cache_hits']} cache hits)")

    usage_path = out_dir / f"{method_name}{score_suffix}_token_usage.json"
    with open(usage_path, "w") as f:
        json.dump({"model": model, "by_split": usage_report}, f, indent=2)
    print(f"  token usage written -> {usage_path.name}")

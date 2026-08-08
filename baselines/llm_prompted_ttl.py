"""
Baseline 1: LLM-prompted TTL.

Prompts a locally served instruct model (via ollama's HTTP API) once per
memory, asking it to output a predicted lifetime in days as a single
number. Responses are parsed to a float; unparseable responses fall back to
a neutral default (7 days). Runs concurrently since ollama can serve
multiple requests in parallel against one loaded model (OLLAMA_NUM_PARALLEL
on the server).

Week 3 used Qwen2.5-7B-Instruct served locally -- see docs/reproducibility.md
for the exact server invocation. Week 4 adds the calibration/consistency/
cost analysis this baseline is really being compared on; this module only
produces the point estimate needed for the Week-3 C-index table.
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from memorylife.utils.io import load_jsonl

FALLBACK_TTL = 7.0
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

PROMPT_TEMPLATE = """You are estimating how long a piece of memory stays useful for a personal AI assistant.

Memory statement: "{text}"

Question: starting from when this was said, how many days from now is this fact likely to remain true and useful to remember, before it becomes outdated or irrelevant? Some facts are permanent (use a large number like 3650 for "essentially forever"), some last weeks or months, and some are only relevant for a day or two.

Answer with ONLY a single integer number of days. No words, no explanation."""


def _query_one(url, model, memory_id, text):
    prompt = PROMPT_TEMPLATE.format(text=text[:500])
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 16},
    }
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            m = NUMBER_RE.search(raw)
            if m:
                return memory_id, max(float(m.group()), 0.01)
            return memory_id, FALLBACK_TTL
        except Exception:
            if attempt == 2:
                return memory_id, FALLBACK_TTL
            time.sleep(2)


def run(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name in args.splits:
        records = load_jsonl(Path(args.data_dir) / f"{split_name}_survival.jsonl")
        scores = {}

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.llm_workers) as ex:
            futures = {
                ex.submit(_query_one, args.llm_url, args.llm_model, r["memory_id"], r["text"]): r["memory_id"]
                for r in records
            }
            for i, fut in enumerate(as_completed(futures), 1):
                mid, val = fut.result()
                scores[mid] = val
                if i % 500 == 0:
                    print(f"  {split_name}: {i}/{len(records)} ({time.time()-t0:.0f}s)")

        with open(out_dir / f"llm_prompted_ttl_{split_name}.json", "w") as f:
            json.dump(scores, f)
        print(f"{split_name}: wrote {len(scores)} scores in {time.time()-t0:.0f}s "
              f"-> llm_prompted_ttl_{split_name}.json")

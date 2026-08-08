"""
Baseline 3: recency-frequency heuristic ("day/week/permanent classifier"'s
untrained sibling).

No training, no embeddings -- a rule-based TTL guess from surface text
alone, the kind of hand-written heuristic a naive memory system would ship
with. Ephemeral keywords (today/tomorrow/deadline/...) push the TTL down;
stable first-person identity/preference statements (my name is/I live
in/...) push it up; everything else gets a generic "week-ish" default,
nudged by text length as a weak proxy for how substantial the statement is.
"""
import json
import re
from pathlib import Path

from memorylife.utils.io import load_jsonl

EPHEMERAL_PAT = re.compile(
    r"\b(today|tonight|tomorrow|this morning|this afternoon|this evening|"
    r"due (on|date)|appointment|meeting|flight|deadline|expir\w*|"
    r"next week|this week)\b",
    re.IGNORECASE,
)
PERMANENT_PAT = re.compile(
    r"\b(my name is|i live in|i am \d|i was born|i have a degree|"
    r"i'm allergic|i am allergic|my favorite|i work as|my job|"
    r"i have \d+ (kids|children|siblings)|permanently|always)\b",
    re.IGNORECASE,
)

BASE_TTL = 14.0
EPHEMERAL_TTL = 2.0
PERMANENT_TTL = 180.0


def score_text(text: str) -> float:
    if EPHEMERAL_PAT.search(text):
        base = EPHEMERAL_TTL
    elif PERMANENT_PAT.search(text):
        base = PERMANENT_TTL
    else:
        base = BASE_TTL
    length_boost = 1.0 + min(len(text) / 200.0, 0.5)
    return base * length_boost


def run(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name in args.splits:
        records = load_jsonl(Path(args.data_dir) / f"{split_name}_survival.jsonl")
        scores = {r["memory_id"]: score_text(r["text"]) for r in records}
        with open(out_dir / f"heuristic_ttl_{split_name}.json", "w") as f:
            json.dump(scores, f)
        print(f"{split_name}: wrote {len(scores)} scores -> heuristic_ttl_{split_name}.json")

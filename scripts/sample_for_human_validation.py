#!/usr/bin/env python
"""
Builds the human-judgment spot-check sample (closes the "no human
validation" Limitations gap) -- separate from and complementary to
scripts/audit_label_consistency.py's automated check, which can only catch
arithmetic bugs, not semantic mistakes (e.g. a plausible-looking but wrong
event_observed call, or an extracted memory that doesn't actually match
what the source dialogue turn says).

THIS SCRIPT DOES NOT DO THE VALIDATION ITSELF. It samples records and, for
LoCoMo records, joins in the actual source dialogue turn so a human can
compare the extracted memory text against what was really said -- but the
"looks correct: yes/no" judgment column is left blank for a real person to
fill in. Reporting agreement statistics from these blanks without a human
having actually filled them in would misrepresent what kind of check was
performed; do not skip that step.

Stratified sample of 45: 15 per source (locomo, longmemeval, synthetic),
split further within each by event_observed=1 vs 0 (censored) to cover
both label types, not just the majority case.

    python scripts/sample_for_human_validation.py
"""
import csv
import json
import random
from pathlib import Path

DATA_DIR = Path("data/processed")
LOCOMO_RAW = Path("data/raw/locomo10.json")
OUT_CSV = Path("results/tables/week7_human_validation_sample.csv")
SEED = 42
N_PER_SOURCE = 15


def load_all_records() -> list[dict]:
    records = []
    for split in ("train", "val", "test"):
        with open(DATA_DIR / f"{split}_survival.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                r["_split"] = split
                records.append(r)
    return records


def build_locomo_turn_lookup() -> dict[str, str]:
    """dia_id -> raw dialogue turn text, across all LoCoMo conversations."""
    data = json.loads(LOCOMO_RAW.read_text(encoding="utf-8"))
    lookup = {}
    for c in data:
        conv = c["conversation"]
        for key, val in conv.items():
            if key.startswith("session_") and not key.endswith("_date_time") and isinstance(val, list):
                for turn in val:
                    lookup[turn["dia_id"]] = f"{turn['speaker']}: {turn['text']}"
    return lookup


def stratified_sample(records: list[dict], rng: random.Random) -> list[dict]:
    sampled = []
    for source in ("locomo", "longmemeval", "synthetic"):
        pool = [r for r in records if r["source"] == source]
        observed = [r for r in pool if r["event_observed"] == 1]
        censored = [r for r in pool if r["event_observed"] == 0]
        n_obs = min(len(observed), N_PER_SOURCE // 2)
        n_cen = min(len(censored), N_PER_SOURCE - n_obs)
        sampled += rng.sample(observed, n_obs) if n_obs else []
        sampled += rng.sample(censored, n_cen) if n_cen else []
    return sampled


def main():
    rng = random.Random(SEED)
    records = load_all_records()
    sample = stratified_sample(records, rng)
    rng.shuffle(sample)  # don't group by source/label -- avoid anchoring bias during review

    locomo_turns = build_locomo_turn_lookup()

    rows = []
    for r in sample:
        source_context = ""
        if r["source"] == "locomo":
            dia_ids = r.get("evidence_dia_id")
            dia_ids = dia_ids if isinstance(dia_ids, list) else [dia_ids]
            source_context = " | ".join(locomo_turns.get(d, f"(dia_id {d} not found)") for d in dia_ids if d)
        elif r["source"] == "synthetic":
            source_context = f"update_text (ground truth): {r.get('update_text', '(none -- record was never updated)')}"
        else:  # longmemeval
            source_context = ("(raw haystack turn text not joined in this sample -- judge from the "
                               "extracted text, timestamps, and lifecycle_event fields only)")

        rows.append({
            "memory_id": r["memory_id"],
            "source": r["source"],
            "extracted_text": r["text"],
            "injected_at": r["injected_at"],
            "lifecycle_event": r.get("lifecycle_event", ""),
            "event_observed": r["event_observed"],
            "censor_reason": r.get("censor_reason", ""),
            "duration_days": round(r["duration_days"], 2),
            "invalidated_at": r.get("invalidated_at") or "",
            "source_context_for_review": source_context,
            "looks_correct_yes_no": "",   # <-- fill this in by hand
            "notes_if_no": "",             # <-- fill this in by hand
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Sampled {len(rows)} records ({sum(1 for r in sample if r['source']=='locomo')} locomo, "
          f"{sum(1 for r in sample if r['source']=='longmemeval')} longmemeval, "
          f"{sum(1 for r in sample if r['source']=='synthetic')} synthetic) "
          f"-> {OUT_CSV}")
    print("\nThis file needs a human to actually open it and fill in "
          "'looks_correct_yes_no' (+ notes) per row before it can be cited as a "
          "human-validation result. Nothing has been pre-filled.")


if __name__ == "__main__":
    main()

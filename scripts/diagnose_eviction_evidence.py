#!/usr/bin/env python
"""
Root-causes the Week-6 finding that `ours` (Lifetime-head TTL expiry +
Action-head "forget") loses to naive `fifo`/`lru` on downstream QA EM/F1
at the same storage budget (results/tables/week6_downstream_qa.md).

Isolates the eviction POLICY from the QA/LLM-answering pipeline: for every
LoCoMo QA pair whose gold evidence was actually extracted as a memory (the
same coverage ceiling `build_benchmark.evidence_coverage` already measures),
checks -- via pure set membership against each policy's final active-memory
set, no LLM calls -- whether at least one evidence-bearing memory survives
eviction. This "evidence retention rate" is a cleaner signal than EM/F1: it
can't be confounded by retrieval top-k, prompt wording, or LLM answering
mistakes, only by which memories the policy chose to keep.

Reuses `run_downstream_qa_eval.compute_lru_last_referenced` and
`final_active_ids` directly (not reimplemented) so the eviction sets here
are guaranteed identical to the ones the real downstream QA eval scored.

For memories that are evidence for some covered QA pair AND get evicted
under `ours`, also attributes each eviction to its mechanism (TTL expiry
vs. Action-head "forget" vs. both) -- tests the hypothesis that the
Action head's known per-class precision gap (forget-class precision 0.635,
update/merge 0.46-0.55; see results/tables/full_comparison_weeks3to5.md)
is why `ours` evicts real evidence more often than a naive recency policy.
Also reports, for the dominant mechanism, the actual TTL-calibration gap
(predicted_ttl_days vs. the age the memory needed to survive to) -- because
C-index (Weeks 3-5's headline metric) is rank-only and invariant to any
monotonic rescaling of the risk score, a model can be an excellent RANKER
of relative memory lifetime while still being systematically miscalibrated
on the ABSOLUTE day-count a TTL-expiry policy actually thresholds against.

    python scripts/diagnose_eviction_evidence.py --device cuda
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_downstream_qa_eval import compute_lru_last_referenced, final_active_ids  # noqa: E402

from memorylife.data.build_benchmark import _dia_ids_of, load_all_processed, load_locomo_qa  # noqa: E402
from memorylife.inference.pipeline import build_memory_objects  # noqa: E402
from memorylife.models.checkpoint import load_joint_model, load_survival_model  # noqa: E402

POLICIES = ("no_forget", "fifo", "lru", "ours", "ours_utility", "ours_combo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--emb-dir", default="artifacts/embeddings")
    ap.add_argument("--feat-dir", default="artifacts/features")
    ap.add_argument("--locomo-path", default="data/raw/locomo10.json")
    ap.add_argument("--survival-model", default="artifacts/survival_model_net.pt")
    ap.add_argument("--joint-model", default="artifacts/joint_model.pt")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ttl-quantile", type=float, default=0.5,
                     help="survival probability S(t) the TTL cutoff crosses; 0.5=median (Week-6 default), "
                          "lower (e.g. 0.2/0.1) pushes the cutoff later -- see quantile_ttl_days docstring")
    ap.add_argument("--out-suffix", default="",
                     help="appended to output filenames, e.g. '_q0.2', so quantile sweeps don't overwrite each other")
    args = ap.parse_args()

    processed = load_all_processed(args.data_dir, args.emb_dir, args.feat_dir)
    feature_slices = json.loads((Path(args.feat_dir) / "feature_slices.json").read_text())

    survival_model = load_survival_model(args.survival_model, 768)
    train_records = [v for v in processed["by_memory_id"].values() if v["split"] == "train"]
    train_emb = np.stack([v["embedding"] for v in train_records]).astype("float32")
    train_dur = np.array([v["record"]["duration_days"] for v in train_records], dtype=np.float32)
    train_ev = np.array([v["record"]["event_observed"] for v in train_records], dtype=np.float32)
    survival_model.compute_baseline_hazards(input=train_emb, target=(train_dur, train_ev))

    joint_model = load_joint_model(args.joint_model)
    joint_model.to(args.device)

    locomo_qa = load_locomo_qa(args.locomo_path)

    retained = {p: [] for p in POLICIES}  # 1/0 per covered QA pair
    eviction_mechanism = Counter()  # for evidence-bearing memories evicted under `ours`
    per_conv_rows = []
    evicted_ttls, evicted_ages, survived_ttls = [], [], []  # TTL-calibration gap, see mechanism table below
    capacities, total_objects = [], []  # storage cost of this quantile choice

    for sample_id, entry in locomo_qa.items():
        answerable_qa = [qa for qa in entry["qa"] if "answer" in qa]
        mids = processed["conversation_to_memory_ids"].get(sample_id)
        if not mids:
            continue
        records = [processed["by_memory_id"][m]["record"] for m in mids]
        embeddings = np.stack([processed["by_memory_id"][m]["embedding"] for m in mids])
        features = np.stack([processed["by_memory_id"][m]["features"] for m in mids])

        objects = build_memory_objects(records, embeddings, features, feature_slices,
                                        survival_model, joint_model, args.device, ttl_quantile=args.ttl_quantile)
        objects_by_id = {o.memory_id: o for o in objects}
        as_of = max(datetime.fromisoformat(r["injected_at"]) for r in records)
        last_referenced = compute_lru_last_referenced(records, embeddings)

        ours_ids = final_active_ids(objects, "ours", None, last_referenced, as_of)
        capacity = len(ours_ids)
        capacities.append(capacity)
        total_objects.append(len(objects))
        active_by_policy = {"ours": ours_ids}
        for policy in ("no_forget", "fifo", "lru", "ours_utility", "ours_combo"):
            active_by_policy[policy] = final_active_ids(objects, policy, capacity, last_referenced, as_of)

        dia_to_mids: dict[str, list[str]] = {}
        for m in mids:
            for d in _dia_ids_of(processed["by_memory_id"][m]["record"]):
                dia_to_mids.setdefault(d, []).append(m)

        conv_covered = 0
        conv_retained = {p: 0 for p in POLICIES}
        for qa in answerable_qa:
            evidence_mids = {m for d in qa["evidence"] for m in dia_to_mids.get(d, [])}
            if not evidence_mids:
                continue  # not covered -- excluded, same ceiling as build_benchmark.evidence_coverage
            conv_covered += 1
            for policy in POLICIES:
                ok = bool(evidence_mids & active_by_policy[policy])
                retained[policy].append(int(ok))
                conv_retained[policy] += int(ok)

            evicted_evidence_mids = evidence_mids - ours_ids
            for mid in evicted_evidence_mids:
                obj = objects_by_id[mid]
                ttl_hit = obj.is_expired(as_of)
                action_hit = obj.action == "forget"
                if ttl_hit and action_hit:
                    eviction_mechanism["both"] += 1
                elif ttl_hit:
                    eviction_mechanism["ttl_only"] += 1
                elif action_hit:
                    eviction_mechanism["action_only"] += 1
                else:
                    eviction_mechanism["neither(?)"] += 1  # shouldn't happen; flags a logic bug if it does
                evicted_ttls.append(obj.predicted_ttl_days)
                evicted_ages.append(obj.age_days(as_of))
            for mid in evidence_mids & ours_ids:
                survived_ttls.append(objects_by_id[mid].predicted_ttl_days)

        if conv_covered:
            per_conv_rows.append({"conversation_id": sample_id, "n_covered_qa": conv_covered,
                                   **{f"{p}_retention": conv_retained[p] / conv_covered for p in POLICIES}})
            print(f"  {sample_id}: {conv_covered} covered QA, retention "
                  + ", ".join(f"{p}={conv_retained[p]/conv_covered:.2f}" for p in POLICIES))

    out_dir = Path(args.out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)

    store_frac = sum(capacities) / sum(total_objects) if total_objects else float("nan")
    summary_lines = [f"TTL quantile (S(t) cutoff): {args.ttl_quantile} "
                      f"(0.5=median). `ours` retains {sum(capacities)}/{sum(total_objects)} "
                      f"memories ({store_frac:.1%}) across all LoCoMo conversations.", "",
                      "| Policy | Evidence retention rate | N covered QA pairs |", "|---|---|---|"]
    for p in POLICIES:
        rate = statistics.mean(retained[p]) if retained[p] else float("nan")
        summary_lines.append(f"| {p} | {rate:.4f} | {len(retained[p])} |")

    mech_total = sum(eviction_mechanism.values())
    mech_lines = ["", "## `ours`-evicted evidence-bearing memories, by mechanism",
                  "", "| Mechanism | Count | % of evicted evidence memories |", "|---|---|---|"]
    for mech in ("ttl_only", "action_only", "both", "neither(?)"):
        c = eviction_mechanism.get(mech, 0)
        pct = (c / mech_total * 100) if mech_total else 0.0
        mech_lines.append(f"| {mech} | {c} | {pct:.1f}% |")

    gap_lines = ["", "## TTL-calibration gap (only meaningful mechanism per the table above)",
                 "", "For each evidence-bearing memory evicted under `ours`, `predicted_ttl_days` "
                 "(Lifetime head) vs. actual age at the point it was still needed as evidence:", ""]
    if evicted_ttls:
        gap = [a - t for a, t in zip(evicted_ages, evicted_ttls)]
        gap_lines += [
            "| | mean | median |", "|---|---|---|",
            f"| predicted_ttl_days (evicted evidence) | {statistics.mean(evicted_ttls):.1f} | {statistics.median(evicted_ttls):.1f} |",
            f"| actual age needed | {statistics.mean(evicted_ages):.1f} | {statistics.median(evicted_ages):.1f} |",
            f"| shortfall (age - predicted_ttl, days) | {statistics.mean(gap):.1f} | {statistics.median(gap):.1f} |",
            f"| predicted_ttl_days (surviving evidence, for reference) | {statistics.mean(survived_ttls):.1f} | {statistics.median(survived_ttls):.1f} |",
        ]

    md = "\n".join(summary_lines + mech_lines + gap_lines)
    table_path = out_dir / "tables" / f"week6_evidence_retention{args.out_suffix}.md"
    raw_path = out_dir / "raw" / f"week6_evidence_retention_per_conv{args.out_suffix}.json"
    table_path.write_text(md, encoding="utf-8")
    raw_path.write_text(json.dumps(per_conv_rows, indent=2))

    print("\n" + md)
    print(f"\nwritten -> {table_path}")


if __name__ == "__main__":
    main()

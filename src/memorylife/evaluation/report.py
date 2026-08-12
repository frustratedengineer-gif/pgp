"""Turns a list of {split, method, c_index, n} rows into the committed
results/tables/*.csv + *.md artifacts."""
from pathlib import Path
import pandas as pd


def write_results_table(rows: list[dict], out_dir: str | Path, name: str = "week3_results_table") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows).sort_values(["split", "c_index"], ascending=[True, False])
    df.to_csv(out_dir / f"{name}.csv", index=False)

    md_lines = ["| Split | Method | C-index | N |", "|---|---|---|---|"]
    for _, r in df.iterrows():
        md_lines.append(f"| {r['split']} | {r['method']} | {r['c_index']:.4f} | {r['n']} |")
    (out_dir / f"{name}.md").write_text("\n".join(md_lines), encoding="utf-8")


def write_token_usage_table(rows: list[dict], out_dir: str | Path, name: str = "llm_token_usage") -> None:
    """rows: {method, model, split, calls, prompt_tokens, completion_tokens,
    total_tokens, avg_tokens_per_call, elapsed_seconds} -- token spend for the
    real-API LLM baselines, reported as its own performance metric alongside
    C-index (cost/efficiency, not accuracy)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows).sort_values(["method", "split"])
    df.to_csv(out_dir / f"{name}.csv", index=False)

    md_lines = [
        "| Method | Model | Split | Calls | Prompt tok | Completion tok | Total tok | Avg tok/call | Time (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        md_lines.append(
            f"| {r['method']} | {r['model']} | {r['split']} | {r['calls']} | "
            f"{r['prompt_tokens']} | {r['completion_tokens']} | {r['total_tokens']} | "
            f"{r['avg_tokens_per_call']} | {r['elapsed_seconds']} |"
        )
    (out_dir / f"{name}.md").write_text("\n".join(md_lines), encoding="utf-8")

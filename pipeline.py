"""
pipeline.py

Main entry point. Runs every (model x task x example) combination N_REPEATS
times, scores each response, computes cost, and writes:
  - results/raw_responses.jsonl   (every single API call's raw output)
  - results/scored_results.csv    (per-example, per-repeat scores + cost + latency)
  - results/summary_by_model_task.csv  (aggregated means/variances for plotting)

Usage:
    python pipeline.py --pilot        # run a small pilot (5 examples/task) to estimate cost
    python pipeline.py                # run the full configured benchmark
    python pipeline.py --models claude-haiku,gpt-frontier --tasks classification
"""

import argparse
import json
import os
from collections import defaultdict

import pandas as pd

from config import MODELS, TASKS, N_REPEATS, RESULTS_DIR, RAW_RESPONSES_FILE, \
    SCORED_RESULTS_FILE, SUMMARY_FILE, PRICING_SNAPSHOT_DATE
from tasks import load_examples
from dispatcher import dispatch
from scorer import score_example
from cost_calculator import query_cost


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", action="store_true",
                    help="Run a tiny pilot (5 examples/task) to sanity-check cost/setup.")
    p.add_argument("--models", type=str, default=None,
                    help="Comma-separated model names to include (default: all in config.py)")
    p.add_argument("--tasks", type=str, default=None,
                    help="Comma-separated task names to include (default: all in config.py)")
    return p.parse_args()


def filter_by_names(items, names_csv, key="name"):
    if not names_csv:
        return items
    wanted = set(n.strip() for n in names_csv.split(","))
    return [it for it in items if getattr(it, key) in wanted]


def run_pipeline(pilot: bool = False, model_filter: str = None, task_filter: str = None):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    models = filter_by_names(MODELS, model_filter)
    tasks = filter_by_names(TASKS, task_filter)

    raw_rows = []
    scored_rows = []

    for task_cfg in tasks:
        n_examples = 5 if pilot else task_cfg.n_examples
        task_cfg.n_examples = n_examples

        print(f"\n=== Loading task: {task_cfg.name} ({n_examples} examples) ===")
        examples = load_examples(task_cfg)

        for model in models:
            print(f"  -> Model: {model.name}")
            for example in examples:
                for repeat_idx in range(1 if pilot else N_REPEATS):
                    result = dispatch(model, example["prompt"])

                    raw_rows.append({
                        "task": task_cfg.name,
                        "model": model.name,
                        "example_id": example["id"],
                        "repeat": repeat_idx,
                        "response_text": result.text,
                        "tokens_in": result.tokens_in,
                        "tokens_out": result.tokens_out,
                        "latency_seconds": result.latency_seconds,
                        "error": result.error,
                        "pricing_snapshot_date": PRICING_SNAPSHOT_DATE,
                    })

                    if result.error:
                        print(f"     [error] {model.name} / {example['id']}: {result.error}")
                        continue

                    cost = query_cost(model, result.tokens_in, result.tokens_out)
                    metric_scores = score_example(task_cfg.name, example, result.text)

                    row = {
                        "task": task_cfg.name,
                        "model": model.name,
                        "tier": model.tier,
                        "provider": model.provider,
                        "example_id": example["id"],
                        "repeat": repeat_idx,
                        "cost_usd": cost,
                        "latency_seconds": result.latency_seconds,
                        "tokens_in": result.tokens_in,
                        "tokens_out": result.tokens_out,
                    }
                    row.update(metric_scores)
                    scored_rows.append(row)

    # --- Persist raw responses (jsonl, append-friendly) ---
    with open(RAW_RESPONSES_FILE, "w") as f:
        for r in raw_rows:
            f.write(json.dumps(r) + "\n")

    # --- Persist per-example scored results ---
    df = pd.DataFrame(scored_rows)
    df.to_csv(SCORED_RESULTS_FILE, index=False)
    print(f"\nWrote {len(df)} scored rows to {SCORED_RESULTS_FILE}")

    # --- Aggregate summary per (model, task) ---
    summarize(df)


def summarize(df: pd.DataFrame):
    if df.empty:
        print("No scored rows to summarize (check for errors above).")
        return

    metric_cols = [c for c in df.columns if c not in
                   {"task", "model", "tier", "provider", "example_id", "repeat",
                    "cost_usd", "latency_seconds", "tokens_in", "tokens_out"}]

    agg_dict = {"cost_usd": ["mean", "std"],
                "latency_seconds": ["mean", "std"],
                "tokens_in": "mean",
                "tokens_out": "mean"}
    for m in metric_cols:
        agg_dict[m] = ["mean", "std"]

    summary = df.groupby(["task", "model", "tier", "provider"]).agg(agg_dict)
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(SUMMARY_FILE, index=False)
    print(f"Wrote summary to {SUMMARY_FILE}")
    print("\n--- Quick preview ---")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(pilot=args.pilot, model_filter=args.models, task_filter=args.tasks)

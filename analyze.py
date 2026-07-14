"""
analyze.py

Reads results/summary_by_model_task.csv and:
  1. Computes the Pareto frontier (cost vs. quality) per task.
  2. Produces scatter plots (cost vs. quality, faceted by task).
  3. Prints a "best value" table (quality / cost ratio) per task.

Usage:
    python analyze.py --quality-metric judge_score_mean --task summarization
    python analyze.py --quality-metric accuracy_mean --task classification
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt

from config import SUMMARY_FILE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--quality-metric", type=str, required=True,
                    help="Column name in summary CSV to use as the quality score, "
                         "e.g. 'accuracy_mean', 'f1_mean', 'judge_score_mean', 'pass_at_1_mean'")
    p.add_argument("--task", type=str, required=True,
                    help="Task name to filter to, e.g. 'summarization'")
    p.add_argument("--output", type=str, default="results/pareto_plot.png")
    return p.parse_args()


def compute_pareto_frontier(df: pd.DataFrame, cost_col: str, quality_col: str):
    """A point is Pareto-optimal if no other point has both lower cost AND
    higher-or-equal quality (i.e., nothing strictly dominates it)."""
    is_pareto = []
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if i == j:
                continue
            better_cost = other[cost_col] <= row[cost_col]
            better_quality = other[quality_col] >= row[quality_col]
            strictly_better = (other[cost_col] < row[cost_col]) or (other[quality_col] > row[quality_col])
            if better_cost and better_quality and strictly_better:
                dominated = True
                break
        is_pareto.append(not dominated)
    df = df.copy()
    df["pareto_optimal"] = is_pareto
    return df


def plot_cost_quality(df: pd.DataFrame, cost_col: str, quality_col: str,
                       task_name: str, output_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in df.iterrows():
        marker = "o" if row["pareto_optimal"] else "x"
        size = 120 if row["pareto_optimal"] else 60
        ax.scatter(row[cost_col], row[quality_col], s=size, marker=marker,
                   label=row["model"])
        ax.annotate(row["model"], (row[cost_col], row[quality_col]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    # Draw the Pareto frontier line connecting optimal points
    pareto_pts = df[df["pareto_optimal"]].sort_values(cost_col)
    ax.plot(pareto_pts[cost_col], pareto_pts[quality_col], linestyle="--",
            color="gray", alpha=0.6, label="Pareto frontier")

    ax.set_xlabel("Average cost per query (USD)")
    ax.set_ylabel(f"Quality ({quality_col})")
    ax.set_title(f"Cost vs. Quality — {task_name}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def main():
    args = parse_args()
    df = pd.read_csv(SUMMARY_FILE)
    df = df[df["task"] == args.task].copy()

    if df.empty:
        print(f"No rows found for task '{args.task}' in {SUMMARY_FILE}")
        return

    cost_col = "cost_usd_mean"
    quality_col = args.quality_metric

    df = compute_pareto_frontier(df, cost_col, quality_col)
    df["value_ratio"] = df[quality_col] / df[cost_col].replace(0, float("nan"))

    print(f"\n=== Task: {args.task} ===")
    print(df[["model", "tier", cost_col, quality_col, "value_ratio", "pareto_optimal"]]
          .sort_values("value_ratio", ascending=False)
          .to_string(index=False))

    plot_cost_quality(df, cost_col, quality_col, args.task, args.output)


if __name__ == "__main__":
    main()

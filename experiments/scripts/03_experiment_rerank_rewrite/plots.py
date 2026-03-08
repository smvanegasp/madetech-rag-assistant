"""Plotting functions for rerank/rewrite experiment results."""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Consistent palette for both plots
METRIC_PALETTE = {"Accuracy": "#2ecc71", "Completeness": "#3498db", "Relevance": "#9b59b6"}


def _add_value_labels(ax, bar_container):
    """Add score labels in the middle of bars."""
    for c in bar_container:
        height = c.get_height()
        if not np.isnan(height):
            ax.annotate(
                f"{height:.2f}",
                xy=(c.get_x() + c.get_width() / 2, height / 2),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
            )


def save_plots(
    output_dir,
    all_results: dict,
    df_summary: pd.DataFrame,
) -> None:
    """
    Save summary plots to the output directory.

    - summary_scores.png: Bar chart of mean accuracy/completeness/relevance by experiment
    - summary_scores_median.png: Bar chart of median accuracy/completeness/relevance by experiment
    - by_question_type.png: Mean scores by question type for the best experiment
    - by_question_type_median.png: Median scores by question type for the best experiment
    """
    output_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir

    # 1. Mean scores by experiment
    df_melt = df_summary.melt(
        id_vars=["experiment"],
        value_vars=["mean_accuracy", "mean_completeness", "mean_relevance"],
        var_name="Metric",
        value_name="Mean Score",
    )
    df_melt["Metric"] = df_melt["Metric"].str.replace("mean_", "").str.capitalize()

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=df_melt,
        x="experiment",
        y="Mean Score",
        hue="Metric",
        palette=METRIC_PALETTE,
        ax=ax,
    )
    for container in ax.containers:
        _add_value_labels(ax, container)
    ax.set_title("Mean scores by experiment configuration", fontsize=13, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    plt.tight_layout()
    fig.savefig(output_dir / "summary_scores.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Median scores by experiment
    median_rows = []
    for name, records in all_results.items():
        acc = [r["accuracy"] for r in records if r.get("accuracy") is not None]
        comp = [r["completeness"] for r in records if r.get("completeness") is not None]
        rel = [r["relevance"] for r in records if r.get("relevance") is not None]
        median_rows.append({
            "experiment": name,
            "median_accuracy": float(np.median(acc)) if acc else np.nan,
            "median_completeness": float(np.median(comp)) if comp else np.nan,
            "median_relevance": float(np.median(rel)) if rel else np.nan,
        })
    df_median = pd.DataFrame(median_rows)
    df_melt_median = df_median.melt(
        id_vars=["experiment"],
        value_vars=["median_accuracy", "median_completeness", "median_relevance"],
        var_name="Metric",
        value_name="Median Score",
    )
    df_melt_median["Metric"] = df_melt_median["Metric"].str.replace("median_", "").str.capitalize()

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=df_melt_median,
        x="experiment",
        y="Median Score",
        hue="Metric",
        palette=METRIC_PALETTE,
        ax=ax,
    )
    for container in ax.containers:
        _add_value_labels(ax, container)
    ax.set_title("Median scores by experiment configuration", fontsize=13, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    plt.tight_layout()
    fig.savefig(output_dir / "summary_scores_median.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Scores by question type (best experiment)
    best_name = df_summary.loc[df_summary["overall_score"].idxmax(), "experiment"]
    best_records = all_results.get(best_name, [])
    if not best_records:
        return

    scores_by_type = defaultdict(lambda: {"accuracy": [], "completeness": [], "relevance": []})
    for r in best_records:
        qt = r.get("question_type", "unknown")
        for m in ["accuracy", "completeness", "relevance"]:
            val = r.get(m)
            if val is not None:
                scores_by_type[qt][m].append(val)

    types = sorted(scores_by_type.keys())
    metrics = ["accuracy", "completeness", "relevance"]
    metric_labels = ["Accuracy", "Completeness", "Relevance"]

    mean_rows = []
    for qt in types:
        for m, label in zip(metrics, metric_labels):
            data = scores_by_type[qt][m]
            mean_score = float(np.mean(data)) if data else np.nan
            mean_rows.append({"Question Type": qt, "Metric": label, "Mean Score": mean_score})
    df_mean = pd.DataFrame(mean_rows)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=df_mean,
        x="Question Type",
        y="Mean Score",
        hue="Metric",
        palette=METRIC_PALETTE,
        capsize=0.1,
        ax=ax,
    )
    for container in ax.containers:
        _add_value_labels(ax, container)
    ax.set_ylabel("Mean Score", fontsize=12)
    ax.set_title(f"Mean Score by Question Type ({best_name})", fontsize=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    plt.tight_layout()
    fig.savefig(output_dir / "by_question_type.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Median scores by question type (best experiment)
    median_rows_by_type = []
    for qt in types:
        for m, label in zip(metrics, metric_labels):
            data = scores_by_type[qt][m]
            median_score = float(np.median(data)) if data else np.nan
            median_rows_by_type.append({"Question Type": qt, "Metric": label, "Median Score": median_score})
    df_median_by_type = pd.DataFrame(median_rows_by_type)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=df_median_by_type,
        x="Question Type",
        y="Median Score",
        hue="Metric",
        palette=METRIC_PALETTE,
        capsize=0.1,
        ax=ax,
    )
    for container in ax.containers:
        _add_value_labels(ax, container)
    ax.set_ylabel("Median Score", fontsize=12)
    ax.set_title(f"Median Score by Question Type ({best_name})", fontsize=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    plt.tight_layout()
    fig.savefig(output_dir / "by_question_type_median.png", dpi=150, bbox_inches="tight")
    plt.close()

"""Plotting functions for the API evaluation experiment results."""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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


def _plot_summary_scores(output_dir: Path, df_summary: pd.DataFrame) -> None:
    """Bar chart of mean accuracy/completeness/relevance by experiment."""
    df_melt = df_summary.melt(
        id_vars=["experiment"],
        value_vars=["mean_accuracy", "mean_completeness", "mean_relevance"],
        var_name="Metric",
        value_name="Mean Score",
    )
    df_melt["Metric"] = df_melt["Metric"].str.replace("mean_", "").str.capitalize()

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_melt, x="experiment", y="Mean Score", hue="Metric", palette=METRIC_PALETTE, ax=ax)
    for container in ax.containers:
        _add_value_labels(ax, container)
    ax.set_title("Mean scores by experiment", fontsize=13, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    plt.tight_layout()
    fig.savefig(output_dir / "summary_scores.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_summary_scores_median(output_dir: Path, all_results: dict) -> None:
    """Bar chart of median accuracy/completeness/relevance by experiment."""
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
    df_melt = df_median.melt(
        id_vars=["experiment"],
        value_vars=["median_accuracy", "median_completeness", "median_relevance"],
        var_name="Metric",
        value_name="Median Score",
    )
    df_melt["Metric"] = df_melt["Metric"].str.replace("median_", "").str.capitalize()

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_melt, x="experiment", y="Median Score", hue="Metric", palette=METRIC_PALETTE, ax=ax)
    for container in ax.containers:
        _add_value_labels(ax, container)
    ax.set_title("Median scores by experiment", fontsize=13, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    plt.tight_layout()
    fig.savefig(output_dir / "summary_scores_median.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_by_question_type(output_dir: Path, all_results: dict, df_summary: pd.DataFrame) -> None:
    """Mean and median scores by question type for the best experiment."""
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

    # Mean
    mean_rows = []
    for qt in types:
        for m, label in zip(metrics, metric_labels):
            data = scores_by_type[qt][m]
            mean_rows.append({"Question Type": qt, "Metric": label, "Mean Score": float(np.mean(data)) if data else np.nan})
    df_mean = pd.DataFrame(mean_rows)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df_mean, x="Question Type", y="Mean Score", hue="Metric", palette=METRIC_PALETTE, ax=ax)
    for container in ax.containers:
        _add_value_labels(ax, container)
    ax.set_title(f"Mean Score by Question Type ({best_name})", fontsize=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    plt.tight_layout()
    fig.savefig(output_dir / "by_question_type.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Median
    median_rows = []
    for qt in types:
        for m, label in zip(metrics, metric_labels):
            data = scores_by_type[qt][m]
            median_rows.append({"Question Type": qt, "Metric": label, "Median Score": float(np.median(data)) if data else np.nan})
    df_median = pd.DataFrame(median_rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df_median, x="Question Type", y="Median Score", hue="Metric", palette=METRIC_PALETTE, ax=ax)
    for container in ax.containers:
        _add_value_labels(ax, container)
    ax.set_title(f"Median Score by Question Type ({best_name})", fontsize=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    plt.tight_layout()
    fig.savefig(output_dir / "by_question_type_median.png", dpi=150, bbox_inches="tight")
    plt.close()


def _build_dist_df(all_results: dict) -> pd.DataFrame:
    """Build long-format DataFrame of per-question scores for distribution plots."""
    rows = []
    for exp_name, records in all_results.items():
        for r in records:
            for m, label in zip(
                ["accuracy", "completeness", "relevance"],
                ["Accuracy", "Completeness", "Relevance"],
            ):
                val = r.get(m)
                if val is not None:
                    rows.append({
                        "experiment": exp_name,
                        "question_type": r.get("question_type", "unknown"),
                        "Metric": label,
                        "Score": val,
                    })
    return pd.DataFrame(rows)


def _plot_histograms(output_dir: Path, df_dist: pd.DataFrame) -> None:
    """Score histograms faceted by experiment and metric."""
    if df_dist.empty:
        return
    metric_order = ["Accuracy", "Completeness", "Relevance"]
    sns.set_theme(style="whitegrid", font_scale=1.1)
    g = sns.FacetGrid(
        df_dist, col="experiment", row="Metric",
        sharex=True, sharey=True, margin_titles=True,
        height=2.5, aspect=1.2, row_order=metric_order,
    )
    g.map_dataframe(sns.histplot, x="Score", bins=range(0, 7), discrete=True, kde=False)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.fig.suptitle("Score histograms by experiment and metric", fontsize=14, fontweight="bold", y=1.02)

    for row_idx, metric in enumerate(metric_order):
        color = METRIC_PALETTE[metric]
        for col_idx in range(g.axes.shape[1]):
            ax = g.axes[row_idx, col_idx]
            for patch in ax.patches:
                patch.set_facecolor(color)
            for patch in ax.patches:
                h = patch.get_height()
                if h > 0:
                    ax.annotate(
                        f"{int(h)}",
                        xy=(patch.get_x() + patch.get_width() / 2, patch.get_y() + h / 2),
                        ha="center", va="center", fontsize=8, fontweight="bold", color="white",
                    )
    plt.tight_layout()
    g.fig.savefig(output_dir / "distribution_histograms.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_ecdf(output_dir: Path, df_dist: pd.DataFrame) -> None:
    """Empirical CDF per experiment."""
    if df_dist.empty:
        return
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)
    for ax, metric in zip(axes, ["Accuracy", "Completeness", "Relevance"]):
        subset = df_dist[df_dist["Metric"] == metric]
        for exp in df_dist["experiment"].unique():
            scores = subset[subset["experiment"] == exp]["Score"].sort_values()
            if len(scores) > 0:
                y = np.arange(1, len(scores) + 1) / len(scores)
                ax.step(np.concatenate([[0], scores, [6]]), np.concatenate([[0], y, [1]]), label=exp, where="post")
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_xlabel("Score")
        ax.set_ylabel("Cumulative proportion")
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", fontsize=7)
    fig.suptitle("Empirical CDF: cumulative proportion at or below each score", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_dir / "distribution_ecdf.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_latency_summary(output_dir: Path, all_results: dict) -> None:
    """Mean, p50, p95 latency bar chart per experiment."""
    rows = []
    for exp_name, records in all_results.items():
        lats = [r["latency_seconds"] for r in records if r.get("latency_seconds") is not None]
        if lats:
            rows.append({"experiment": exp_name, "Stat": "Mean", "Latency (s)": float(np.mean(lats))})
            rows.append({"experiment": exp_name, "Stat": "p50", "Latency (s)": float(np.median(lats))})
            rows.append({"experiment": exp_name, "Stat": "p95", "Latency (s)": float(np.percentile(lats, 95))})
    if not rows:
        return
    df_lat = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_lat, x="experiment", y="Latency (s)", hue="Stat",
                palette={"Mean": "#e74c3c", "p50": "#f39c12", "p95": "#c0392b"}, ax=ax)
    for container in ax.containers:
        for c in container:
            h = c.get_height()
            if not np.isnan(h) and h > 0:
                ax.annotate(f"{h:.1f}s", xy=(c.get_x() + c.get_width() / 2, h / 2),
                            ha="center", va="center", fontsize=8, fontweight="bold", color="white")
    ax.set_title("API latency by experiment (end-to-end)", fontsize=13, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    plt.tight_layout()
    fig.savefig(output_dir / "latency_summary.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_failure_rates(output_dir: Path, all_results: dict) -> None:
    """API vs judge error rates per experiment."""
    rows = []
    for exp_name, records in all_results.items():
        total = len(records)
        api_err = sum(1 for r in records if r.get("api_error"))
        judge_err = sum(1 for r in records if r.get("judge_error"))
        if total > 0:
            rows.append({"experiment": exp_name, "Failure Type": "API error", "Rate (%)": round(api_err / total * 100, 1)})
            rows.append({"experiment": exp_name, "Failure Type": "Judge error", "Rate (%)": round(judge_err / total * 100, 1)})
    if not rows:
        return
    df_fail = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_fail, x="experiment", y="Rate (%)", hue="Failure Type",
                palette={"API error": "#e74c3c", "Judge error": "#f39c12"}, ax=ax)
    for container in ax.containers:
        for c in container:
            h = c.get_height()
            if not np.isnan(h) and h > 0:
                ax.annotate(f"{h:.1f}%", xy=(c.get_x() + c.get_width() / 2, h / 2),
                            ha="center", va="center", fontsize=8, fontweight="bold", color="white")
    ax.set_title("Failure rates by experiment", fontsize=13, fontweight="bold")
    ax.set_ylabel("Failure Rate (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    plt.tight_layout()
    fig.savefig(output_dir / "failure_rates.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_latency_distribution(output_dir: Path, all_results: dict) -> None:
    """Box plot of latency distribution per experiment."""
    rows = []
    for exp_name, records in all_results.items():
        for r in records:
            lat = r.get("latency_seconds")
            if lat is not None:
                rows.append({"experiment": exp_name, "Latency (s)": lat})
    if not rows:
        return
    df = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="experiment", y="Latency (s)", color="#e74c3c", fliersize=3, ax=ax)
    ax.set_title("API latency distribution by experiment", fontsize=13, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "latency_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


# =========================================================================
# Cross-run comparison plots
# =========================================================================

def _plot_comparison_scores(output_dir: Path, combined_summary: pd.DataFrame) -> None:
    """Grouped bar chart comparing mean scores across multiple runs."""
    if combined_summary.empty or len(combined_summary) < 2:
        return
    df_melt = combined_summary.melt(
        id_vars=["experiment"],
        value_vars=["mean_accuracy", "mean_completeness", "mean_relevance"],
        var_name="Metric",
        value_name="Mean Score",
    )
    df_melt["Metric"] = df_melt["Metric"].str.replace("mean_", "").str.capitalize()

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(max(10, len(combined_summary) * 2.5), 5))
    sns.barplot(data=df_melt, x="experiment", y="Mean Score", hue="Metric", palette=METRIC_PALETTE, ax=ax)
    for container in ax.containers:
        _add_value_labels(ax, container)
    ax.set_title("Cross-run comparison: Mean scores", fontsize=13, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    plt.tight_layout()
    fig.savefig(output_dir / "comparison_scores.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_comparison_latency(output_dir: Path, combined_summary: pd.DataFrame) -> None:
    """Latency comparison across multiple runs."""
    if combined_summary.empty or len(combined_summary) < 2:
        return
    rows = []
    for _, row in combined_summary.iterrows():
        name = row["experiment"]
        rows.append({"experiment": name, "Stat": "Mean", "Latency (s)": row.get("mean_latency_s", 0)})
        rows.append({"experiment": name, "Stat": "p50", "Latency (s)": row.get("p50_latency_s", 0)})
        rows.append({"experiment": name, "Stat": "p95", "Latency (s)": row.get("p95_latency_s", 0)})
    df_lat = pd.DataFrame(rows)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(max(10, len(combined_summary) * 2.5), 5))
    sns.barplot(data=df_lat, x="experiment", y="Latency (s)", hue="Stat",
                palette={"Mean": "#e74c3c", "p50": "#f39c12", "p95": "#c0392b"}, ax=ax)
    for container in ax.containers:
        for c in container:
            h = c.get_height()
            if not np.isnan(h) and h > 0:
                ax.annotate(f"{h:.1f}s", xy=(c.get_x() + c.get_width() / 2, h / 2),
                            ha="center", va="center", fontsize=8, fontweight="bold", color="white")
    ax.set_title("Cross-run comparison: API latency", fontsize=13, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    plt.tight_layout()
    fig.savefig(output_dir / "comparison_latency.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_comparison_overall(output_dir: Path, combined_summary: pd.DataFrame) -> None:
    """Overall score comparison bar chart across runs."""
    if combined_summary.empty or len(combined_summary) < 2:
        return
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(max(8, len(combined_summary) * 2), 5))
    bars = sns.barplot(data=combined_summary, x="experiment", y="overall_score", color="#2ecc71", ax=ax)
    for c in bars.containers:
        for patch in c:
            h = patch.get_height()
            if not np.isnan(h) and h > 0:
                ax.annotate(f"{h:.2f}", xy=(patch.get_x() + patch.get_width() / 2, h / 2),
                            ha="center", va="center", fontsize=10, fontweight="bold", color="white")
    ax.set_title("Cross-run comparison: Overall score", fontsize=13, fontweight="bold")
    ax.set_ylabel("Overall Score (avg of accuracy, completeness, relevance)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "comparison_overall.png", dpi=150, bbox_inches="tight")
    plt.close()


# =========================================================================
# Main entry point
# =========================================================================

def save_plots(
    output_dir,
    all_results: dict,
    df_summary: pd.DataFrame,
    compare_results: dict | None = None,
    compare_summary: pd.DataFrame | None = None,
) -> None:
    """
    Save all plots to the output directory.

    Single-run plots:
    - summary_scores.png: Mean accuracy/completeness/relevance by experiment
    - summary_scores_median.png: Median scores by experiment
    - by_question_type.png / by_question_type_median.png: Scores by question type
    - distribution_histograms.png: Score histograms faceted by experiment
    - distribution_ecdf.png: Empirical CDF per experiment
    - latency_summary.png: Mean/p50/p95 latency
    - failure_rates.png: API vs judge error rates
    - latency_distribution.png: Latency box plot

    Cross-run comparison plots (when compare_with data is provided):
    - comparison_scores.png: Mean scores across runs
    - comparison_latency.png: Latency across runs
    - comparison_overall.png: Overall score across runs
    """
    output_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir

    # Single-run plots
    _plot_summary_scores(output_dir, df_summary)
    _plot_summary_scores_median(output_dir, all_results)
    _plot_by_question_type(output_dir, all_results, df_summary)

    df_dist = _build_dist_df(all_results)
    _plot_histograms(output_dir, df_dist)
    _plot_ecdf(output_dir, df_dist)
    _plot_latency_summary(output_dir, all_results)
    _plot_failure_rates(output_dir, all_results)
    _plot_latency_distribution(output_dir, all_results)

    print(f"Saved plots to {output_dir}")

    # Cross-run comparison plots
    if compare_results and compare_summary is not None and not compare_summary.empty:
        combined_results = {**compare_results, **all_results}
        combined_summary = pd.concat([compare_summary, df_summary], ignore_index=True)
        # Deduplicate by experiment name (keep latest = current run)
        combined_summary = combined_summary.drop_duplicates(subset=["experiment"], keep="last")

        _plot_comparison_scores(output_dir, combined_summary)
        _plot_comparison_latency(output_dir, combined_summary)
        _plot_comparison_overall(output_dir, combined_summary)

        # Also generate comparison distribution plots (ECDF with all runs)
        df_dist_combined = _build_dist_df(combined_results)
        if not df_dist_combined.empty:
            _plot_ecdf(output_dir, df_dist_combined)
            output_dir_cmp = output_dir
            # Rename the ECDF to avoid overwriting single-run version
            ecdf_path = output_dir / "distribution_ecdf.png"
            ecdf_cmp_path = output_dir / "comparison_ecdf.png"
            if ecdf_path.exists():
                ecdf_path.rename(ecdf_cmp_path)
                # Re-generate single-run ECDF
                df_dist_single = _build_dist_df(all_results)
                _plot_ecdf(output_dir, df_dist_single)

        print(f"Saved cross-run comparison plots to {output_dir}")

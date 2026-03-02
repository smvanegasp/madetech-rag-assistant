"""Plotting functions for eval question analysis."""

import logging
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Defined here directly to avoid a cross-module bare import that would break
# if this file is loaded before sys.path includes the script directory.
SCORE_DIMS = ["relevance", "standalone", "groundedness"]

PALETTE = {"Before filter": "#4472C4", "After filter": "#ED7D31"}


def _build_before_after_df(
    df_all: pd.DataFrame, df_filtered: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine all records and filtered records into one DataFrame with a 'split'
    label column ('Before filter' / 'After filter').

    The two DataFrames are stacked vertically so seaborn can use 'split' as
    the hue dimension in a single barplot call.
    """
    df_b = df_all[["doc_category"] + SCORE_DIMS].copy()
    df_b["split"] = "Before filter"
    df_a = df_filtered[["doc_category"] + SCORE_DIMS].copy()
    df_a["split"] = "After filter"
    return pd.concat([df_b, df_a], ignore_index=True)


def _polish_bars(ax) -> None:
    """Add white bar edges and a subtle dashed horizontal grid for readability."""
    for patch in ax.patches:
        patch.set_edgecolor("white")
        patch.set_linewidth(0.8)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.6, color="#CCCCCC")
    ax.set_axisbelow(True)


def plot_before_after_bar(
    df_all: pd.DataFrame,
    df_filtered: pd.DataFrame,
    output_path: Path,
    quality_cutoff: int = 4,
) -> None:
    """
    Grouped bar chart: mean score per dimension, before vs after the quality filter.
    Error bars show +/-1 SD.
    """
    logger.debug("Building before/after bar chart -> %s", output_path)

    combined = _build_before_after_df(df_all, df_filtered)
    # Melt from wide (one column per dimension) to long format so seaborn can
    # map dimension -> x-axis and split -> hue in a single call.
    melted = combined.melt(
        id_vars=["doc_category", "split"], var_name="dimension", value_name="score"
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=melted,
        x="dimension",
        y="score",
        hue="split",
        palette=PALETTE,
        capsize=0.04,
        errorbar="sd",
        ax=ax,
        err_kws={"linewidth": 1.2},
    )
    _polish_bars(ax)
    # Red dashed line marks the quality cutoff so viewers can immediately see
    # how the filter threshold relates to the actual score distribution.
    ax.axhline(
        y=quality_cutoff,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label=f"Quality cutoff ({quality_cutoff})",
        zorder=3,
    )
    ax.set_ylim(0, 5.5)
    ax.set_title(
        "Mean Score per Dimension — Before vs After Quality Filter",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Mean Score")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="", frameon=False)

    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.debug("Saved -> %s", output_path)


def plot_category_before_after_bar(
    df_all: pd.DataFrame,
    df_filtered: pd.DataFrame,
    output_path: Path,
    quality_cutoff: int = 4,
) -> None:
    """
    One subplot per critique dimension; each shows mean score by document category,
    before vs after the quality filter. Error bars show +/-1 SD.
    """
    logger.debug("Building category before/after bar chart -> %s", output_path)

    combined = _build_before_after_df(df_all, df_filtered)
    melted = combined.melt(
        id_vars=["doc_category", "split"], var_name="dimension", value_name="score"
    )

    fig, axes = plt.subplots(1, len(SCORE_DIMS), figsize=(15, 5), sharey=True)
    fig.suptitle(
        "Mean Score by Category — Before vs After Quality Filter",
        fontsize=13,
        fontweight="bold",
    )

    for ax, dim in zip(axes, SCORE_DIMS):
        # Filter to one dimension at a time so each subplot stays focused.
        subset = melted[melted["dimension"] == dim]
        sns.barplot(
            data=subset,
            x="doc_category",
            y="score",
            hue="split",
            palette=PALETTE,
            capsize=0.04,
            errorbar="sd",
            ax=ax,
            err_kws={"linewidth": 1.2},
        )
        _polish_bars(ax)
        ax.axhline(
            y=quality_cutoff,
            color="crimson",
            linestyle="--",
            linewidth=1.2,
            zorder=3,
        )
        ax.set_title(dim.capitalize(), fontsize=11)
        ax.set_xlabel("Category")
        # Only label y-axis on the leftmost subplot; sharey=True links the axes.
        ax.set_ylabel("Mean Score" if ax is axes[0] else "")
        ax.set_ylim(0, 5.5)
        ax.tick_params(axis="x", rotation=20)
        ax.get_legend().remove()

    # Single shared legend for the whole figure using manual Patch handles.
    legend_elements = [
        mpatches.Patch(
            facecolor=PALETTE["Before filter"],
            edgecolor="white",
            linewidth=0.8,
            label="Before filter",
        ),
        mpatches.Patch(
            facecolor=PALETTE["After filter"],
            edgecolor="white",
            linewidth=0.8,
            label="After filter",
        ),
    ]
    fig.legend(handles=legend_elements, loc="upper right", frameon=False)

    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.debug("Saved -> %s", output_path)


def plot_scores_distribution(
    df: pd.DataFrame, output_path: Path, quality_cutoff: int = 4
) -> None:
    """
    Histogram of score distribution (1–5) for each critique dimension.

    A vertical dashed line at quality_cutoff - 0.5 shows the pass/fail boundary.
    """
    logger.debug("Building scores distribution histogram -> %s", output_path)

    # Melt so each row represents one (dimension, score) observation.
    melted = df[SCORE_DIMS].melt(var_name="dimension", value_name="score")

    fig, axes = plt.subplots(1, len(SCORE_DIMS), figsize=(12, 4), sharey=True)
    fig.suptitle("Score Distribution by Dimension", fontsize=13, fontweight="bold")

    for ax, dim in zip(axes, SCORE_DIMS):
        subset = melted[melted["dimension"] == dim]
        sns.histplot(
            data=subset,
            x="score",
            bins=range(1, 7),
            discrete=True,
            ax=ax,
            stat="count",
        )
        # Place the cutoff line between bins (cutoff - 0.5) so it sits clearly
        # on the boundary rather than overlapping a bar.
        ax.axvline(
            x=quality_cutoff - 0.5,
            color="crimson",
            linestyle="--",
            linewidth=1.2,
            label="Quality cutoff",
            zorder=3,
        )
        ax.set_title(dim.capitalize(), fontsize=11)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count" if ax is axes[0] else "")
        ax.set_xticks([1, 2, 3, 4, 5])

    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.debug("Saved -> %s", output_path)

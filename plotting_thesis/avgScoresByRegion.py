import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from pathlib import Path
import os

def load_and_average(categories_path, evaluations, regions, results_dir):
    """
    Loads category and RSA evaluation CSVs, computes average scores per region.
    Returns a DataFrame with regions as rows and evaluation types as columns.
    """
    categories_df = pd.read_csv(categories_path)

    avg_scores = {region: [] for region in regions}

    for eval_name in evaluations:
        score_df = pd.read_csv(results_dir / f"{eval_name}.csv")
        merged_df = pd.merge(categories_df, score_df, on="Model", how="inner")

        for region in regions:
            avg_score = merged_df[f"%R2_{region}"].mean(skipna=True)
            avg_scores[region].append(avg_score)

    return pd.DataFrame(avg_scores, index=evaluations).T


def plot_grouped_bar(avg_scores_df, evaluations, regions, legend_map):
    """
    Creates and saves a grouped bar plot of average RSA explained variance per region.
    """
    x = np.arange(len(regions))  # x-axis positions
    width = 0.8 / len(evaluations)  # dynamically scale width based on number of bars
    cmap = cm.get_cmap("viridis", len(evaluations))

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, eval_name in enumerate(evaluations):
        bars = ax.bar(
            x + i * width,
            avg_scores_df[eval_name],
            width,
            label=legend_map.get(eval_name, eval_name),
            color=cmap(i),
        )

        # Optional: Add value labels above bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # offset
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    ax.set_ylabel("Explained Variance RSA in %", fontsize=14)
    ax.set_xticks(x + width * (len(evaluations) - 1) / 2)
    ax.set_xticklabels(regions)
    ax.legend(loc="upper left")

    plt.tight_layout()
    output_dir = f"../plots/thesis"
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(f"{output_dir}/barplot/avgScoresByRegion.png")
    plt.show()


if __name__ == "__main__":
    # Configuration
    results_dir = Path("../results")

    evaluations = ["rsa_natural", "rsa_imagenet", "rsa_synthetic", "rsa_illusion"]
    regions = ["V1", "V2", "V4", "IT"]

    legend_map = {
        "rsa_illusion": "Kamitani Illusion",
        "rsa_imagenet": "Kamitani ImageNet",
        "rsa_synthetic": "NSD Synthetic",
        "rsa_natural": "NSD Natural",
    }

    avg_scores_df = load_and_average(
        categories_path=results_dir / "categories.csv",
        evaluations=evaluations,
        regions=regions,
        results_dir=results_dir,
    )

    plot_grouped_bar(avg_scores_df, evaluations, regions, legend_map)

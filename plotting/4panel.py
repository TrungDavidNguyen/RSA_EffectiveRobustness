import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import linregress
from matplotlib.lines import Line2D

def create_plot(ood_dataset, roi, evaluation, all_models=False, ax=None):
    roi_name = f"%R2_{roi}" if "rsa" in evaluation else f"R_{roi}"

    # Load data
    brain_similarity = pd.read_csv(f"../results/{evaluation}.csv").dropna(subset=[roi_name])
    robustness = pd.read_csv("../results/effective_robustness.csv")
    categories = pd.read_csv("../results/categories.csv")
    architectures = categories["architecture"].unique()

    # Merge
    df = pd.merge(brain_similarity, robustness, on="Model", how="inner")
    df = pd.merge(df, categories, on="Model", how="inner")

    if not all_models:
        df = df[df["architecture"] == "CNN"]

    if ood_dataset == "imagenet-a":
        df = df[df["Model"].str.lower() != "resnet50"].reset_index(drop=True)

    # Marker & color maps
    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>']
    datasets = categories["dataset"].unique()
    marker_map = {ds: markers[i % len(markers)] for i, ds in enumerate(datasets)}

    colors = plt.cm.tab10.colors
    color_map = {arch: colors[i % len(colors)] for i, arch in enumerate(architectures)}

    # Use provided axes or create a new one
    if ax is None:
        fig, ax = plt.subplots()

    # Scatter points
    for _, row in df.iterrows():
        ax.scatter(
            row[roi_name],
            row[ood_dataset],
            marker=marker_map[row["dataset"]],
            color=color_map[row["architecture"]],
            edgecolor="black",
            s=50,
        )

    # Regression line
    slope, intercept, r_value, p_value, _ = linregress(df[roi_name], df[ood_dataset])
    x_vals = df[roi_name]
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, color="red")

    # Correlation box
    ax.text(
        0.05,
        0.95,
        f"r = {r_value:.2f}\np = {p_value:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    ax.set_title(roi, fontsize=12)

    return ax


if __name__ == "__main__":
    evaluations = ["encoding_natural"]
    ood_datasets = ["imagenet-r"]
    rois = ["V1", "V2", "V4", "IT"]

    for ood_dataset in ood_datasets:
        for evaluation in evaluations:
            # Create 2x2 figure
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # no sharex/sharey
            axes = axes.flatten()

            for i, roi in enumerate(rois):
                create_plot(ood_dataset, roi, evaluation, ax=axes[i])
            pretty_eval_name = evaluation.replace("rsa", "RSA").replace("encoding", "Encoding")

            # Global axis labels
            fig.text(0.5, 0.04, pretty_eval_name, ha="center", fontsize=12)
            fig.text(0.04, 0.5, f"Effective robustness to {ood_dataset}", va="center", rotation="vertical", fontsize=12)

            plt.tight_layout(rect=[0.05, 0.05, 1, 1])
            output_dir = f"../plots/thesis"
            os.makedirs(output_dir, exist_ok=True)

            plt.savefig(f"{output_dir}/all_ROIs_{ood_dataset}_{evaluation}.png")
            plt.show()


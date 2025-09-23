import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import linregress
import plotting_thesis.utils as utils

def create_plot(ood_dataset, roi, evaluation, all_models=False):
    roi_name = f"%R2_{roi}" if "rsa" in evaluation else f"R_{roi}"
    roi_names = utils.PlottingConfig.ROIS

    # Load data
    brain_similarity = pd.read_csv(f"../results/{evaluation}.csv").dropna(
        subset=[utils.get_roi_col_name(roi, evaluation) for roi in roi_names])
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

    # Scatter points
    for _, row in df.iterrows():
        plt.scatter(
            row[roi_name],
            row[ood_dataset],
            marker=marker_map[row["dataset"]],
            color=color_map[row["architecture"]],
            edgecolor="black",
            s=50,
        )
        plt.text(row[roi_name], row[ood_dataset], row["Model"],
                 fontsize=7, ha='right', va='bottom')

    # Regression line
    slope, intercept, r_value, p_value, _ = linregress(df[roi_name], df[ood_dataset])
    x_vals = df[roi_name]
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color="red")

    # Legend
    """    plt.text(0.05, 0.95, f"r = {r_value:.2f}\np = {p_value:.2f}",
                 transform=plt.gca().transAxes,
                 ha='left', va='top',
                 fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
        dataset_handles = [Line2D([0], [0], marker=marker_map[ds], color='w', label=ds,
                                  markerfacecolor='gray', markersize=8, markeredgecolor='black')
                           for ds in datasets]
    
        architecture_handles = [Line2D([0], [0], marker='o', color=color_map[arch], label=arch,
                                       linestyle='None', markersize=8)
                                for arch in architectures]
    
        legend1 = plt.legend(handles=dataset_handles, title="Dataset (Shape)", loc='upper right', fontsize=6, title_fontsize=8)
        plt.gca().add_artist(legend1)
        plt.legend(handles=architecture_handles, title="Architecture (Color)", loc='lower right', fontsize=6, title_fontsize=8)"""

    # Correlation box
    plt.text(
        0.05,
        0.95,
        f"r = {r_value:.2f}\np = {p_value:.2f}",
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    # Axis labels
    pretty_eval_name = evaluation.replace("rsa", "RSA").replace("encoding", "Encoding")
    #plt.xlabel(pretty_eval_name, fontsize=15)
    #plt.ylabel(f"Effective Robustness to {ood_dataset}", fontsize=15)
    plt.title(roi)

    # Save
    model_type = "all models" if all_models else "only CNNs"
    plt.tight_layout()
    output_dir = f"../plots/brain_vs_rob/{evaluation}/{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{roi}_{ood_dataset}_{evaluation}.png")
    plt.close()


if __name__ == "__main__":
    evaluations = utils.PlottingConfig.EVALUATIONS
    ood_datasets = ["imagenet-r", "imagenet-sketch", "imagenetv2-matched-frequency", "imagenet-a"]
    rois = ["V1", "V2", "V4", "IT"]

    for ood_dataset in ood_datasets:
        for roi in rois:
            for evaluation in evaluations:
                create_plot(ood_dataset, roi, evaluation, True)
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import linregress
from matplotlib.lines import Line2D


def create_plot(ood_dataset, roi, evaluation, all_models = False):
    eval_name = f"%R2_{evaluation}" if "rsa" in evaluation else f"R_{evaluation}"
    roi_name = f"%R2_{roi}" if "rsa" in evaluation else f"R_{roi}"

    brain_similarity = pd.read_csv(f"../results/{evaluation}.csv")
    robustness = pd.read_csv("../results/effective_robustness.csv")
    categories = pd.read_csv("../results/categories.csv")
    architectures = categories["architecture"].unique()

    brain_similarity = brain_similarity.dropna(subset=[roi_name])

    df = pd.merge(brain_similarity, robustness, on='Model', how='inner')
    df = pd.merge(df, categories, on='Model', how='inner')
    if not all_models:
        df = df[(df["dataset"] != "more data") & (df["architecture"] == "CNN")]

    if ood_dataset == "imagenet-a":
        df = df[df['Model'].str.lower() != "resnet50"]
        df = df.reset_index(drop=True)

    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>']
    datasets = df["dataset"].unique()
    marker_map = {ds: markers[i % len(markers)] for i, ds in enumerate(datasets)}

    colors = plt.cm.tab10.colors
    color_map = {arch: colors[i % len(colors)] for i, arch in enumerate(architectures)}

    for _, row in df.iterrows():
        plt.scatter(row[roi_name], row[ood_dataset],
                    marker=marker_map[row["dataset"]],
                    color=color_map[row["architecture"]],
                    edgecolor='black',
                    s=50,
                    label=f'{row["dataset"]}_{row["architecture"]}')  # Temporary for deduplication

        plt.text(row[roi_name], row[ood_dataset], row["Model"],
                 fontsize=7, ha='right', va='bottom')

    slope, intercept, r_value, p_value, std_err = linregress(df[roi_name], df[ood_dataset])
    x_vals = df[roi_name]
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color="red")

    plt.text(0.05, 0.95, f"r = {r_value:.2f}\np = {p_value:.2f}",
             transform=plt.gca().transAxes,
             ha='left', va='top',
             fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    dataset_handles = [Line2D([0], [0], marker=marker_map[ds], color='w', label=ds,
                              markerfacecolor='gray', markersize=8, markeredgecolor='black')
                       for ds in datasets]

    architecture_handles = [Line2D([0], [0], marker='o', color=color_map[arch], label=arch,
                                   linestyle='None', markersize=8)
                            for arch in architectures]

    """    legend1 = plt.legend(handles=dataset_handles, title="Dataset (Shape)", loc='upper right', fontsize=6, title_fontsize=8)
        plt.gca().add_artist(legend1)
        plt.legend(handles=architecture_handles, title="Architecture (Color)", loc='lower right', fontsize=6, title_fontsize=8)"""
    model_type = "all_models" if all_models else "only_CNNs_imagenet1k"

    plt.xlabel(eval_name)
    plt.ylabel("Effective Robustness")
    plt.title(f"{roi} and {ood_dataset}")
    plt.tight_layout()
    output_dir = f"../plots/brain_vs_rob/{evaluation}/{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{roi}_{ood_dataset}_{evaluation}.png")
    #plt.show()
    plt.close()


if __name__ == '__main__':
    evaluations = [
        "encoding_natural", "rsa_natural",
        "encoding_synthetic", "rsa_synthetic",
        "encoding_illusion", "rsa_illusion",
        "encoding_imagenet", "rsa_imagenet"
    ]
    ood_datasets = ["imagenet-r", "imagenet-sketch", "imagenetv2-matched-frequency", "imagenet-a"]
    rois = ["V1", "V2", "V4", "IT"]
    for ood_dataset in ood_datasets:
        for roi in rois:
            for evaluation in evaluations:
                create_plot(ood_dataset, roi, evaluation)
                create_plot(ood_dataset, roi, evaluation, True)
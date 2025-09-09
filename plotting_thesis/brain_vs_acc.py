import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import linregress
from matplotlib.lines import Line2D


def create_plot(dataset, roi, evaluation, all_models=False):
    eval_name = f"%R2_{evaluation}" if "rsa" in evaluation else f"R_{evaluation}"
    roi_name = f"%R2_{roi}" if "rsa" in evaluation else f"R_{roi}"
    brain_similarity = pd.read_csv(f"../results/{evaluation}.csv")
    robustness = pd.read_csv("../results/accuracies.csv")
    categories = pd.read_csv("../results/categories.csv")
    # --- global stable lists ---
    all_architectures = sorted(categories["architecture"].unique())
    all_datasets = sorted(categories["dataset"].unique())

    brain_similarity = brain_similarity.dropna(subset=[roi_name])

    df = pd.merge(brain_similarity, robustness, on='Model', how='inner')
    df = pd.merge(df, categories, on='Model', how='inner')
    if not all_models:
        df = df[(df["dataset"] != "more data") & (df["architecture"] == "CNN")]

    if dataset == "imagenet-a":
        df = df[df['Model'].str.lower() != "resnet50"]
        df = df.reset_index(drop=True)

    # --- stable mappings ---
    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>']
    marker_map = {ds: markers[i % len(markers)] for i, ds in enumerate(all_datasets)}

    colors = plt.cm.tab10.colors
    color_map = {arch: colors[i % len(colors)] for i, arch in enumerate(all_architectures)}

    for _, row in df.iterrows():
        plt.scatter(row[roi_name], row[dataset],
                    marker=marker_map[row["dataset"]],
                    color=color_map[row["architecture"]],
                    edgecolor='black',
                    s=50,
                    label=f'{row["dataset"]}_{row["architecture"]}')

    """        plt.text(row[roi_name], row[dataset], row["Model"],
                     fontsize=7, ha='right', va='bottom')"""
    # regression
    slope, intercept, r_value, p_value, std_err = linregress(df[roi_name], df[dataset])
    x_vals = df[roi_name]
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color="red")

    plt.text(0.05, 0.95, f"r = {r_value:.2f}\np = {p_value:.2f}",
             transform=plt.gca().transAxes,
             ha='left', va='top',
             fontsize=15, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))


    architecture_handles = [Line2D([0], [0], marker='o', color=color_map[arch], label=arch,
                                   linestyle='None', markersize=8)
                            for arch in all_architectures]


    plt.legend(handles=architecture_handles, title="Architecture (Color)",
               loc='lower left', fontsize=10, title_fontsize=12)

    model_type = "all_models" if all_models else "only_CNNs_imagenet1k"

    plt.xlabel("Encoding R on V1 of Illusion dataset",fontsize=17)
    plt.ylabel(f"Accuracy on ImageNet-1K", fontsize=17)
    plt.title(f"Only CNNs",fontsize=22)
    plt.tight_layout()

    output_dir = f"../plots/brain_vs_acc/{evaluation}/{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{roi}_{dataset}_{evaluation}.png")
    plt.close()



if __name__ == '__main__':
    evaluations = [
        "encoding_illusion", "rsa_illusion",
    ]
    ood_datasets = ["imagenet1k"]
    rois = ["V1", "V2", "V4", "IT"]
    for ood_dataset in ood_datasets:
        for roi in rois:
            for evaluation in evaluations:
                create_plot(ood_dataset, roi, evaluation)
                create_plot(ood_dataset, roi, evaluation, True)
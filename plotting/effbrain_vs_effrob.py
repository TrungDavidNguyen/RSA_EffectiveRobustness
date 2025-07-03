import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import linregress
from matplotlib.lines import Line2D


def create_plot(dataset, roi, evaluation, all_models=False):
    if "rsa" in evaluation:
        eval_name = f"%R2_{roi}_{evaluation}"
    elif "encoding" in evaluation:
        eval_name = f"R_{roi}_{evaluation}"

    brain_similarity = pd.read_csv(f"../results/effective_brain_similarity.csv")
    robustness = pd.read_csv(f"../results/effective_robustness.csv")
    categories = pd.read_csv("../results/categories.csv")

    df = pd.merge(brain_similarity, robustness, on='Model', how='inner')
    df = pd.merge(df, categories, on='Model', how='inner')
    if not all_models:
        df = df[(df["dataset"] != "more data") & (df["architecture"] == "CNN")]
    df = df.dropna()
    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>']
    datasets = df["dataset"].unique()
    marker_map = {ds: markers[i % len(markers)] for i, ds in enumerate(datasets)}

    architectures = df["architecture"].unique()
    colors = plt.cm.tab10.colors
    color_map = {arch: colors[i % len(colors)] for i, arch in enumerate(architectures)}

    for _, row in df.iterrows():
        plt.scatter(row[eval_name], row[dataset],
                    marker=marker_map[row["dataset"]],
                    color=color_map[row["architecture"]],
                    edgecolor='black',
                    s=50,
                    label=f'{row["dataset"]}_{row["architecture"]}')

        plt.text(row[eval_name], row[dataset], row["Model"],
                 fontsize=7, ha='right', va='bottom')

    # Regression line
    slope, intercept, r_value, p_value, std_err = linregress(df[eval_name], df[dataset])
    x_vals = df[dataset]
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color="red")

    plt.text(0.95, 0.95, f"r = {r_value:.2f}\np = {p_value:.2f}",
             transform=plt.gca().transAxes,
             ha='right', va='top',
             fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    # Create custom legend entries for dataset (markers)
    dataset_handles = [Line2D([0], [0], marker=marker_map[ds], color='w', label=ds,
                              markerfacecolor='gray', markersize=8, markeredgecolor='black')
                       for ds in datasets]

    architecture_handles = [Line2D([0], [0], marker='o', color=color_map[arch], label=arch,
                                   linestyle='None', markersize=8)
                            for arch in architectures]

    legend1 = plt.legend(handles=dataset_handles, title="Dataset (Shape)", loc='lower left', fontsize=6, title_fontsize=8)
    plt.gca().add_artist(legend1)
    plt.legend(handles=architecture_handles, title="Architecture (Color)", loc='lower right', fontsize=6, title_fontsize=8)
    model_type = "all_models" if all_models else "only_CNNs_imagenet1k"
    plt.title(model_type)
    plt.xlabel(f"effective brain similarity {evaluation} {roi}")
    plt.ylabel(f"effective robustness {dataset}")
    plt.tight_layout()
    output_dir = f"../plots/effbrain_vs_effrob/{evaluation}/{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir,f"{ood_dataset}_{evaluation}_{roi}"))
    plt.close()


if __name__ == '__main__':
    for ood_dataset in ["imagenet-r","imagenet-sketch", "imagenetv2-matched-frequency","imagenet-a"]:
        for roi in ["V1", "V2", "V4","IT"]:
            for eval in ["encoding_synthetic","encoding_illusion","rsa_synthetic","rsa_illusion"]:
                create_plot(ood_dataset, roi, eval)
                create_plot(ood_dataset, roi, eval, True)
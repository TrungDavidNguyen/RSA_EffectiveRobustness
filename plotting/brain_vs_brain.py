import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import linregress
from matplotlib.lines import Line2D


def create_plot(roi, id, ood, all_models=False):
    roi_name_id = f"%R2_{roi}" if "rsa" in id else f"R_{roi}"
    roi_name_ood = f"%R2_{roi}" if "rsa" in ood else f"R_{roi}"

    df_id = pd.read_csv(f"../results/{id}.csv")

    df_ood = pd.read_csv(f"../results/{ood}.csv")
    cols_to_rename = {col: f"{col}_{ood}" for col in df_ood.columns if
                      col != 'Model'}
    df_ood = df_ood.rename(columns=cols_to_rename)

    categories = pd.read_csv("../results/categories.csv")

    df = pd.merge(df_id, df_ood, on='Model', how='inner')
    df = pd.merge(df, categories, on='Model', how='inner')
    df.dropna()
    if not all_models:
        df = df[df["architecture"] == "CNN"]

    """    if "encoding" in id:
            df[roi_name] = logit(df[roi_name]*100)
            df[f"{roi_name}_{ood}"] = logit(df[f"{roi_name}_{ood}"]*100)
        else:
            df[roi_name] = logit(df[roi_name])
            df[f"{roi_name}_{ood}"] = logit(df[f"{roi_name}_{ood}"])"""

    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>']
    datasets = df["dataset"].unique()
    marker_map = {ds: markers[i % len(markers)] for i, ds in enumerate(datasets)}

    architectures = df["architecture"].unique()
    colors = plt.cm.tab10.colors
    color_map = {arch: colors[i % len(colors)] for i, arch in enumerate(architectures)}
    for _, row in df.iterrows():
        plt.scatter(row[roi_name_id], row[f"{roi_name_ood}_{ood}"],
                    marker=marker_map[row["dataset"]],
                    color=color_map[row["architecture"]],
                    edgecolor='black',
                    s=50,
                    label=f'{row["dataset"]}_{row["architecture"]}')

        plt.text(row[roi_name_id], row[f"{roi_name_ood}_{ood}"], row["Model"],
                 fontsize=7, ha='right', va='bottom')

    slope, intercept, r_value, p_value, std_err = linregress(df[roi_name_id], df[f"{roi_name_ood}_{ood}"])
    x_vals = df[roi_name_id]
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color="red")

    plt.text(0.95, 0.95, f"r = {r_value:.2f}\np = {p_value:.2f}",
             transform=plt.gca().transAxes,
             ha='right', va='top',
             fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    dataset_handles = [Line2D([0], [0], marker=marker_map[ds], color='w', label=ds,
                              markerfacecolor='gray', markersize=8, markeredgecolor='black')
                       for ds in datasets]

    architecture_handles = [Line2D([0], [0], marker='o', color=color_map[arch], label=arch,
                                   linestyle='None', markersize=8)
                            for arch in architectures]

    legend1 = plt.legend(handles=dataset_handles, title="Dataset (Shape)", loc='lower left', fontsize=6, title_fontsize=8)
    plt.gca().add_artist(legend1)
    plt.legend(handles=architecture_handles, title="Architecture (Color)", loc='lower right', fontsize=6, title_fontsize=8)
    model_type = "all models" if all_models else "only CNNs"

    plt.xlabel(f"{id} {roi_name_id}")
    plt.ylabel(f"{ood} {roi_name_ood}")
    plt.title(f"{id}_vs_{ood}_{roi} {model_type}")
    plt.tight_layout()
    output_dir = f"../plots/brain_vs_brain/{model_type}/{roi}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{id}_vs_{ood}_{roi}"))
    plt.show()


if __name__ == '__main__':
    evaluations = [
        "encoding_imagenet", "rsa_imagenet",
        "encoding_natural", "rsa_natural",
        "encoding_synthetic", "rsa_synthetic",
        "encoding_illusion", "rsa_illusion"
    ]
    create_plot("IT", "rsa_imagenet", "rsa_illusion")

    for roi in ["V1", "V2", "V4", "IT"]:
        for i, eval in enumerate(evaluations):
            for eval_2 in evaluations[i+1:]:
                create_plot(roi, eval, eval_2)
                create_plot(roi, eval, eval_2, True)

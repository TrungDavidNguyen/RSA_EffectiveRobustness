import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import linregress
from effective_robustness import logit


def create_plot(roi, id, ood, all_models=False):
    roi_name = f"%R2_{roi}" if "rsa" in id else f"R_{roi}"

    df_id = pd.read_csv(f"../results/{id}.csv")
    df_id = df_id.dropna(subset=[roi_name])

    df_ood = pd.read_csv(f"../results/{ood}.csv")

    cols_to_rename = {col: f"{col}_{ood}" for col in df_ood.columns if
                      col != 'Model' and col in df_id.columns}
    brain_similarity_synth_renamed = df_ood.rename(columns=cols_to_rename)
    categories = pd.read_csv("../results/categories.csv")

    df = pd.merge(df_id, brain_similarity_synth_renamed, on='Model', how='inner')
    df = pd.merge(df, categories, on='Model', how='inner')
    if not all_models:
        df = df[(df["dataset"] != "more data") & (df["architecture"] == "CNN")]

    """    if "encoding" in id:
            df[roi_name] = logit(df[roi_name]*100)
            df[f"{roi_name}_{ood}"] = logit(df[f"{roi_name}_{ood}"]*100)
        else:
            df[roi_name] = logit(df[roi_name])
            df[f"{roi_name}_{ood}"] = logit(df[f"{roi_name}_{ood}"])"""

    # Define distinct markers for datasets
    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>']
    datasets = df["dataset"].unique()
    marker_map = {ds: markers[i % len(markers)] for i, ds in enumerate(datasets)}

    # Define a color map for architecture (optional)
    architectures = df["architecture"].unique()
    colors = plt.cm.tab10.colors
    color_map = {arch: colors[i % len(colors)] for i, arch in enumerate(architectures)}

    # Plot points with marker by dataset and color by architecture
    for _, row in df.iterrows():
        plt.scatter(row[roi_name], row[f"{roi_name}_{ood}"],
                    marker=marker_map[row["dataset"]],
                    color=color_map[row["architecture"]],
                    edgecolor='black',
                    s=50,
                    label=f'{row["dataset"]}_{row["architecture"]}')  # Temporary for deduplication

        plt.text(row[roi_name], row[f"{roi_name}_{ood}"], row["Model"],
                 fontsize=7, ha='right', va='bottom')

    # Regression line
    slope, intercept, r_value, p_value, std_err = linregress(df[roi_name], df[f"{roi_name}_{ood}"])
    print("slope", slope)
    print("intercept", intercept)
    x_vals = df[roi_name]
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color="red")

    # Correlation label
    plt.text(0.95, 0.95, f"r = {r_value:.2f}\np = {p_value:.2f}",
             transform=plt.gca().transAxes,
             ha='right', va='top',
             fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    # Create custom legend entries for dataset (markers)
    from matplotlib.lines import Line2D
    dataset_handles = [Line2D([0], [0], marker=marker_map[ds], color='w', label=ds,
                              markerfacecolor='gray', markersize=8, markeredgecolor='black')
                       for ds in datasets]

    architecture_handles = [Line2D([0], [0], marker='o', color=color_map[arch], label=arch,
                                   linestyle='None', markersize=8)
                            for arch in architectures]

    legend1 = plt.legend(handles=dataset_handles, title="Dataset (Shape)", loc='lower left', fontsize=6, title_fontsize=8)
    plt.gca().add_artist(legend1)
    plt.legend(handles=architecture_handles, title="Architecture (Color)", loc='lower right', fontsize=6, title_fontsize=8)

    plt.xlabel(f"{id} {roi_name}")
    plt.ylabel(f"{ood} {roi_name}")
    plt.title(f"{id}_vs_{ood}_{roi}")
    plt.tight_layout()
    os.makedirs("../plots/brain_similarity", exist_ok=True)
    plt.savefig(f"../plots/brain_similarity/{id}_vs_{ood}_{roi}")
    plt.show()


if __name__ == '__main__':
    for roi in ["V1", "V2", "V4", "IT"]:
        create_plot(roi, "encoding_natural", "encoding_illusion")
        create_plot(roi, "encoding_natural", "encoding_illusion", True)


import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from matplotlib.lines import Line2D


def create_plot(ood_dataset, roi, evaluation):
    eval_name = f"%R2_{evaluation}" if evaluation in ["rsa", "rsa_synthetic"] else f"R_{evaluation}"
    roi_name = f"%R2_{roi}" if evaluation in ["rsa", "rsa_synthetic"] else f"R_{roi}"

    brain_similarity = pd.read_csv(f"results/{evaluation}.csv")
    robustness = pd.read_csv("results/effective_robustness.csv")
    categories = pd.read_csv("results/categories.csv")
    architectures = categories["architecture"].unique()

    brain_similarity = brain_similarity.dropna(subset=[roi_name])

    df = pd.merge(brain_similarity, robustness, on='Model', how='inner')
    df = pd.merge(df, categories, on='Model', how='inner')
    #df = df[df["dataset"] != "more data"]

    if ood_dataset == "imagenet-a":
        df = df[df['Model'].str.lower() != "resnet50"]
        df = df.reset_index(drop=True)

    #df = df[df["dataset"] != "more data"]
    #df = df[df["architecture"] != "VIT"]

    # Define distinct markers for datasets
    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>']
    datasets = df["dataset"].unique()
    marker_map = {ds: markers[i % len(markers)] for i, ds in enumerate(datasets)}

    # Define a color map for architecture (optional)
    colors = plt.cm.tab10.colors
    color_map = {arch: colors[i % len(colors)] for i, arch in enumerate(architectures)}

    # Plot points with marker by dataset and color by architecture
    for _, row in df.iterrows():
        plt.scatter(row[roi_name], row[ood_dataset],
                    marker=marker_map[row["dataset"]],
                    color=color_map[row["architecture"]],
                    edgecolor='black',
                    s=50,
                    label=f'{row["dataset"]}_{row["architecture"]}')  # Temporary for deduplication

        plt.text(row[roi_name], row[ood_dataset], row["Model"],
                 fontsize=7, ha='right', va='bottom')

    # Regression line
    slope, intercept, r_value, p_value, std_err = linregress(df[roi_name], df[ood_dataset])
    x_vals = df[roi_name]
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

    #legend1 = plt.legend(handles=dataset_handles, title="Dataset (Shape)", loc='upper right', fontsize=6, title_fontsize=8)
    #plt.gca().add_artist(legend1)
    #plt.legend(handles=architecture_handles, title="Architecture (Color)", loc='lower right', fontsize=6, title_fontsize=8)

    plt.xlabel(eval_name)
    plt.ylabel("Effective Robustness")
    plt.title(f"{roi} and {ood_dataset}")
    plt.tight_layout()
    plt.savefig(f"plots/{roi}_{ood_dataset}_{evaluation}")
    plt.show()


if __name__ == '__main__':
    for ood_dataset in ["imagenet-r","imagenet-sketch", "imagenetv2-matched-frequency","imagenet-a"]:
        for roi in ["V1", "V2", "V4","IT"]:
            create_plot(ood_dataset, roi, "encoding_synthetic")
            create_plot(ood_dataset, roi, "encoding")
            create_plot(ood_dataset, roi, "rsa")
            create_plot(ood_dataset, roi, "rsa_synthetic")


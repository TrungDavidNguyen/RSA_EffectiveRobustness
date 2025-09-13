import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress


def create_bar_plot_by_model(method, rois, model=None):
    df = pd.read_csv(f"../results/{method}.csv")
    metric = "%R2" if method.startswith("rsa") else "R"

    df_plot = df.set_index('Model')
    columns = [f"{metric}_{roi}" for roi in rois]
    df_plot = df_plot[columns]

    if model is not None:
        df_plot = df_plot.loc[[model]]

    fig, ax = plt.subplots(figsize=(16, 6))
    df_plot.plot(kind='bar', ax=ax)

    ax.set_xlabel('Model')
    ax.set_ylabel(metric)
    ax.set_title(f'{method} {metric} results per Model')
    ax.legend(title=f'{method}/ROI')

    plt.tight_layout()
    os.makedirs("../plots/barplot", exist_ok=True)
    plt.savefig(f"../plots/barplot/barplot_by_model_{method}.png")
    plt.show()


def create_bar_plot_by_roi(method, roi, models=None):
    df = pd.read_csv(f"../results/{method}.csv")
    categories = pd.read_csv("../results/categories.csv")

    architectures = categories["architecture"].unique()
    colors = plt.cm.tab10.colors
    color_map = {arch: colors[i % len(colors)] for i, arch in enumerate(architectures)}

    metric = "%R2" if method.startswith("rsa") else "R"
    column_name = f"{metric}_{roi}"

    # Merge with architecture info
    df = df.merge(categories[['Model', 'architecture']], on='Model', how='left')

    # Filter for specific models if provided
    if models is not None:
        df = df[df['Model'].isin(models)]

    # Sort by architecture and score
    #df = df.sort_values(by=['architecture', column_name], ascending=[True, False])
    bar_colors = df['architecture'].map(color_map)

    plt.figure(figsize=(10, 6))
    plt.bar(df['Model'], df[column_name], color=bar_colors)

    plt.xlabel('Model')
    plt.ylabel(f'{metric} value')
    plt.title(f'Method: {method} – ROI: {roi}')
    plt.xticks(rotation=45, ha='right')

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[arch]) for arch in architectures]
    plt.legend(handles, architectures, title='Architecture')

    plt.tight_layout()
    os.makedirs("../plots/barplot", exist_ok=True)
    plt.savefig(f"../plots/barplot/barplot_{roi}_{method}.png")
    plt.show()


def corr(method, rois):
    df_nat = pd.read_csv(f"../results/{method}.csv")
    df_syn = pd.read_csv(f"../results/{method}_synthetic.csv")

    merged = pd.merge(df_nat, df_syn, on='Model', suffixes=(f'_{method}', f'_{method}_synthetic'))
    metric = "%R2" if method.startswith("rsa") else "R"

    x_col = f"{metric}_{rois[0]}_{method}_synthetic"
    y_col = f"{metric}_{rois[0]}_{method}"

    if x_col not in merged.columns or y_col not in merged.columns:
        print(f"Missing columns for ROI {rois[0]}")
        return

    merged = merged.dropna(subset=[x_col, y_col])

    slope, intercept, r_value, p_value, std_err = linregress(merged[x_col], merged[y_col])
    print(f"{method} correlation for ROI {rois[0]}: r = {r_value:.3f}, p = {p_value:.3e}")


if __name__ == '__main__':
    # Example usage:
    #create_bar_plot_by_model("rsa_synthetic", ["V1", "V2", "V4", "IT"])
    datasets = ["rsa_natural","rsa_imagenet", "rsa_synthetic", "rsa_illusion"]
    #datasets = ["encoding_natural","encoding_imagenet", "encoding_synthetic", "encoding_illusion"]

    models = ["Densenet121", "Densenet161", "Densenet169", "Densenet201"]
    #models = ["ResNet18", "ResNet34", "ResNet50", "ResNet101","ResNet152"]
    #models = ["VGG11", "VGG13", "VGG16", "VGG19"]
    models = ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4", "efficientnet_b5"]

    for roi in ["V1", "V2", "V4", "IT"]:
        for method in datasets:
            create_bar_plot_by_roi(method, roi, models=models)

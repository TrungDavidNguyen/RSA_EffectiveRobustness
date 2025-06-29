import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress


def logit(acc):
    return np.log(np.divide(acc / 100.0, 1.0 - acc / 100.0))

def inv_logit(acc):
    return (np.exp(acc)/(1 + np.exp(acc)))*100

def logit_fit(id_dataset, ood_dataset):

    accuracies = pd.read_csv("../results/accuracies.csv")
    categories = pd.read_csv("../results/categories.csv")

    df = pd.merge(accuracies, categories, on='Model', how='inner')
    """    if ood_dataset == "imagenet-a":
            df = df[df['Model'].str.lower() != "resnet50"]
            df = df.reset_index(drop=True)"""

    df = df[df["architecture"] == "CNN"]
    df = df[df["dataset"] == "ImageNet1K"]
    # Define a color map for architecture (optional)
    architectures = df["architecture"].unique()
    colors = plt.cm.tab10.colors
    color_map = {arch: colors[i % len(colors)] for i, arch in enumerate(architectures)}

    # Plot points with marker by dataset and color by architecture
    for _, row in df.iterrows():
        plt.scatter(row[id_dataset], row[ood_dataset],
                    color=color_map[row["architecture"]],
                    edgecolor='black',
                    s=50,
                    label=f'{row["dataset"]}_{row["architecture"]}')  # Temporary for deduplication

        plt.text(row[id_dataset], row[ood_dataset], row["Model"],
                 fontsize=7, ha='right', va='bottom')    # Regression line
    slope, intercept, r_value, p_value, std_err = linregress(logit(df[id_dataset]), logit(df[ood_dataset]))
    plt.text(0.05, 0.95, f"r = {r_value:.2f}\np = {p_value:.2f}",
             transform=plt.gca().transAxes,
             ha='left', va='top',
             fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    # Create a smooth range of x values within the valid (0,1) interval
    x_vals = np.linspace(df[id_dataset].min(), df[id_dataset].max(), 100)
    x_logit = logit(x_vals)
    y_vals = inv_logit(slope * x_logit + intercept)
    plt.title("ID vs OOD-Accuracy")
    plt.xlabel(id_dataset)
    plt.ylabel(ood_dataset)
    plt.plot(x_vals, y_vals, color="red", linewidth=2)
    output_dir = f"../plots/id_vs_ood"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{id_dataset}_vs_{ood_dataset}.png")
    plt.show()
    return slope, intercept

def linear_fit(id_dataset, ood_dataset):
    accuracies = pd.read_csv("../results/accuracies.csv")
    categories = pd.read_csv("../results/categories.csv")

    df = pd.merge(accuracies, categories, on='Model', how='inner')
    """    if ood_dataset == "imagenet-a":
            df = df[df['Model'].str.lower() != "resnet50"]
            df = df.reset_index(drop=True)"""

    df = df[df["architecture"] == "CNN"]
    df = df[df["dataset"] == "ImageNet1K"]
    # Define a color map for architecture (optional)
    architectures = df["architecture"].unique()
    colors = plt.cm.tab10.colors
    color_map = {arch: colors[i % len(colors)] for i, arch in enumerate(architectures)}

    # Plot points with marker by dataset and color by architecture
    for _, row in df.iterrows():
        plt.scatter(row[id_dataset], row[ood_dataset],
                    color=color_map[row["architecture"]],
                    edgecolor='black',
                    s=50,
                    label=f'{row["dataset"]}_{row["architecture"]}')  # Temporary for deduplication

        plt.text(row[id_dataset], row[ood_dataset], row["Model"],
                 fontsize=7, ha='right', va='bottom')    # Regression line
    slope, intercept, r_value, p_value, std_err = linregress(df[id_dataset], df[ood_dataset])
    plt.text(0.05, 0.95, f"r = {r_value:.2f}\np = {p_value:.2f}",
             transform=plt.gca().transAxes,
             ha='left', va='top',
             fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    # Create a smooth range of x values within the valid (0,1) interval
    x_vals = np.linspace(df[id_dataset].min(), df[id_dataset].max(), 100)
    y_vals = slope * x_vals + intercept
    plt.title("ID vs OOD-Accuracy")
    plt.xlabel(id_dataset)
    plt.ylabel(ood_dataset)
    plt.plot(x_vals, y_vals, color="red", linewidth=2)

    plt.show()
    return slope, intercept


if __name__ == '__main__':
    datasets = {"imagenet-r": "imagenet1k-subset-r",
                "imagenet-sketch": "imagenet1k",
                "imagenetv2-matched-frequency": "imagenet1k",
                "imagenet-a": "imagenet1k-subset-a"}
    for ood, id in datasets.items():
        logit_fit(id,ood)
    for ood, id in datasets.items():
        linear_fit(id, ood)
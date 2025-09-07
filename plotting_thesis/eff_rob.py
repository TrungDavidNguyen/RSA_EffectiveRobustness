import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress
import seaborn as sns

sns.set(style="whitegrid")  # prettier style

def logit(acc):
    return np.log(np.divide(acc / 100.0, 1.0 - acc / 100.0))

def inv_logit(acc):
    return (np.exp(acc)/(1 + np.exp(acc)))*100

def logit_fit(id_dataset, ood_dataset):
    accuracies = pd.read_csv("../results/accuracies.csv")
    categories = pd.read_csv("../results/categories.csv")

    df = pd.merge(accuracies, categories, on='Model', how='inner')
    df = df[df["architecture"] == "CNN"]
    df = df[df["dataset"] == "ImageNet1K"]

    # Set up colors
    architectures = df["architecture"].unique()
    palette = sns.color_palette("tab10", n_colors=len(architectures))
    color_map = {arch: palette[i] for i, arch in enumerate(architectures)}

    plt.figure(figsize=(10, 7))
    plotted_labels = set()

    for _, row in df.iterrows():
        label = f'{row["dataset"]}_{row["architecture"]}'
        # Avoid duplicate labels in legend
        if label not in plotted_labels:
            plt.scatter(row[id_dataset], row[ood_dataset],
                        color=color_map[row["architecture"]],
                        s=70,
                        label=label)
            plotted_labels.add(label)
        else:
            plt.scatter(row[id_dataset], row[ood_dataset],
                        color=color_map[row["architecture"]],
                        s=70)

        plt.text(row[id_dataset], row[ood_dataset], row["Model"],
                 fontsize=8, ha='right', va='bottom', alpha=0.7)

    slope, intercept, r_value, p_value, std_err = linregress(logit(df[id_dataset]), logit(df[ood_dataset]))

    plt.text(0.05, 0.95, f"r = {r_value:.2f}\np = {p_value:.3f}",
             transform=plt.gca().transAxes,
             ha='left', va='top',
             fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    x_vals = np.linspace(df[id_dataset].min(), df[id_dataset].max(), 100)
    x_logit = logit(x_vals)
    y_vals = inv_logit(slope * x_logit + intercept)

    plt.plot(x_vals, y_vals, color="red", linewidth=2, label="Logit fit")
    plt.plot(x_vals, x_vals, 'k--', label='y = x')

    plt.title(f"ID vs OOD Accuracy ({id_dataset} vs {ood_dataset})", fontsize=16)
    plt.xlabel("id_dataset", fontsize=14)
    plt.ylabel("ood_dataset", fontsize=14)
    plt.legend(frameon=True, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_dir = "../plots/id_vs_ood"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{id_dataset}_vs_{ood_dataset}.png", dpi=300)
    plt.show()

    return slope, intercept

def linear_fit(id_dataset, ood_dataset):
    accuracies = pd.read_csv("../results/accuracies.csv")
    categories = pd.read_csv("../results/categories.csv")

    df = pd.merge(accuracies, categories, on='Model', how='inner')
    df = df[df["architecture"] == "CNN"]
    df = df[df["dataset"] == "ImageNet1K"]

    architectures = df["architecture"].unique()
    palette = sns.color_palette("tab10", n_colors=len(architectures))
    color_map = {arch: palette[i] for i, arch in enumerate(architectures)}

    plt.figure(figsize=(10, 6))
    plotted_labels = set()

    for _, row in df.iterrows():
        label = f'Standard CNN'
        if label not in plotted_labels:
            plt.scatter(row[id_dataset], row[ood_dataset],
                        color=color_map[row["architecture"]],
                        s=70,
                        label=label)
            plotted_labels.add(label)
        else:
            plt.scatter(row[id_dataset], row[ood_dataset],
                        color=color_map[row["architecture"]],
                        s=70)

    slope, intercept, r_value, p_value, std_err = linregress(df[id_dataset], df[ood_dataset])

    x_vals = np.linspace(df[id_dataset].min(), df[id_dataset].max(), 100)
    y_vals = slope * x_vals + intercept

    # Example new point
    x_new = 70  # Example ID accuracy
    y_new = 64  # Example OOD accuracy

    # Predicted y from baseline
    y_baseline = slope * x_new + intercept

    # Plot the new point in green
    plt.scatter(x_new, y_new, color='purple', s=100, edgecolor='black', zorder=5, label='example model')

    # Add arrow from baseline to the new point
    plt.annotate(
        '', xy=(x_new, y_new), xytext=(x_new, y_baseline),
        arrowprops=dict(facecolor='purple', shrink=0.05, width=3, headwidth=8),
        zorder=4
    )

    plt.text(x_new - 7, (y_new + y_baseline) / 2, "Effective Robustness",
             fontsize=13, fontweight='bold', color='purple', ha='left', va='center')

    plt.plot(x_vals, y_vals, color="orange", linewidth=2, label="baseline")
    plt.plot(x_vals, x_vals, 'k--', label='y = x')

    plt.title(f"ID vs OOD Accuracy", fontsize=20, fontweight='bold')
    plt.xlabel("id accuracy (%)", fontsize=16, fontweight='bold')
    plt.ylabel("ood accuracy (%)", fontsize=16, fontweight='bold')
    plt.legend(frameon=True, framealpha=0.9, fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_dir = f"../plots/thesis"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"../plots/thesis/eff_rob.png", dpi=300)
    plt.show()

    return slope, intercept


if __name__ == '__main__':
    datasets = {"imagenetv2-matched-frequency": "imagenet1k"}

    for ood, id in datasets.items():
        linear_fit(id, ood)

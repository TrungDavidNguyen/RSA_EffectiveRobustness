import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D


def performance_jitter_plot(data_file, categories_file):
    # Load data
    df = pd.read_csv(data_file)
    categories = pd.read_csv(categories_file)
    print(categories.head())
    categories = categories[categories["dataset"] == "ImageNet1K"]
    print(categories.head())
    df = df.merge(categories, on='Model', how='inner')

    # Melt dataframe to long format
    df_long = df.melt(
        id_vars=["Model", "architecture"],
        value_vars=["imagenet-r", "imagenet-sketch", "imagenet-a", "imagenetv2-matched-frequency"],
        var_name="dataset",
        value_name="performance"
    )

    datasets = df_long['dataset'].unique()
    architectures = df_long['architecture'].unique()

    # Generate colors dynamically for all architectures
    cmap = cm.get_cmap('tab10', len(architectures))
    arch_colors = {arch: cmap(i) for i, arch in enumerate(architectures)}

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, dataset in enumerate(datasets, start=1):
        n_arch = len(architectures)
        offsets = np.linspace(-0.2, 0.2, n_arch)  # spread architectures for jitter

        for arch, offset in zip(architectures, offsets):
            scores = df_long.loc[(df_long['architecture'] == arch) & (df_long['dataset'] == dataset), 'performance']

            # Jitter x positions
            x_jitter = np.random.normal(i + offset, 0.03, size=len(scores))
            ax.scatter(x_jitter, scores, color=arch_colors[arch], alpha=0.6, edgecolor='k', linewidth=0.3)

            # Plot mean
            mean_val = scores.mean()
            ax.plot([i + offset - 0.05, i + offset + 0.05], [mean_val, mean_val], color=arch_colors[arch], lw=2)
            print(f"{dataset} - {arch} mean: {mean_val:.3f}")

    # Beautify plot
    ax.set_xticks(range(1, len(datasets) + 1))
    ax.set_xticklabels(datasets, rotation=30, fontsize=12)
    ax.set_ylabel("Performance", fontsize=14)
    ax.set_xlabel("Dataset", fontsize=14)
    ax.set_title("Model Performance by Dataset & Architecture", fontsize=16, weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Custom legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=arch,
                              markerfacecolor=color, markersize=10)
                       for arch, color in arch_colors.items()]
    ax.legend(handles=legend_elements, frameon=False, fontsize=12, title='Architecture')

    plt.tight_layout()

    # Save plot
    output_file = "../plots/performance_jitter_plot_all_arch.png"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    plt.show()


if __name__ == '__main__':
    performance_jitter_plot("../results/effective_robustness.csv", "../results/categories.csv")

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from scipy.stats import ttest_ind


def performance_jitter_plot(data_file, categories_file):
    # Load data
    df = pd.read_csv(data_file)
    categories = pd.read_csv(categories_file)

    # Filter out architectures you don’t want
    categories = categories[~categories["architecture"].isin(["MLP", "RNN"])]
    #categories = categories[categories["dataset"] != "ImageNet1K"]

    df = df.merge(categories, on='Model', how='inner')

    # ✅ Rename the column before melting
    df = df.rename(columns={"imagenetv2-matched-frequency": "imagenetv2"})

    # Melt dataframe to long format
    df_long = df.melt(
        id_vars=["Model", "architecture"],
        value_vars=["imagenet-r", "imagenet-sketch", "imagenet-a", "imagenetv2"],
        var_name="dataset",
        value_name="performance"
    )

    datasets = df_long['dataset'].unique()
    architectures = df_long['architecture'].unique()

    # Generate colors dynamically for architectures
    cmap = cm.get_cmap('viridis', 2)
    arch_colors = {arch: cmap(i) for i, arch in enumerate(architectures)}

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, dataset in enumerate(datasets, start=1):
        n_arch = len(architectures)
        offsets = np.linspace(-0.25, 0.25, n_arch)  # spread architectures

        arch_means = {}

        for arch, offset in zip(architectures, offsets):
            scores = df_long.loc[
                (df_long['architecture'] == arch) & (df_long['dataset'] == dataset),
                'performance'
            ]

            # Store mean for later comparison
            arch_means[arch] = scores.mean()

            # Jitter scatter
            x_jitter = np.random.normal(i + offset, 0.03, size=len(scores))
            ax.scatter(x_jitter, scores, color=arch_colors[arch],
                       alpha=0.7, edgecolor='k', linewidth=0.3)

            # Mean horizontal line
            ax.plot([i + offset - 0.08, i + offset + 0.08],
                    [scores.mean(), scores.mean()],
                    color=arch_colors[arch], lw=2)

        # Perform pairwise Welch’s t-tests between architectures
        for j in range(len(architectures)):
            for k in range(j + 1, len(architectures)):
                arch1, arch2 = architectures[j], architectures[k]
                scores1 = df_long.loc[
                    (df_long['architecture'] == arch1) & (df_long['dataset'] == dataset),
                    'performance'
                ]
                scores2 = df_long.loc[
                    (df_long['architecture'] == arch2) & (df_long['dataset'] == dataset),
                    'performance'
                ]
                t_stat, p_val = ttest_ind(scores1, scores2, equal_var=False)
                diff = scores1.mean() - scores2.mean()
                print(f"[{dataset}] {arch1} vs {arch2}: "
                      f"T={t_stat:.3f}, p={p_val:.3g}, "
                      f"Mean1={scores1.mean():.3f}, Mean2={scores2.mean():.3f}, Diff={diff:.3f}")

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
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              label=arch, markerfacecolor=color, markersize=10)
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

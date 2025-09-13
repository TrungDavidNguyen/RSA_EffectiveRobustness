import numpy as np
import pandas as pd
import os
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from matplotlib import cm

def jitter_plot(method, regions):
    df = pd.read_csv(f"../results/{method}.csv")
    categories = pd.read_csv("../results/categories.csv")
    df = df.merge(categories, on='Model', how='left')
    df.dropna()
    # Get two colors from viridis colormap
    cmap = cm.get_cmap('viridis', 2)
    cnn_color = cmap(0)  # one end of the colormap
    vit_color = cmap(1)  # the other end

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, region in enumerate(regions, start=1):
        score = f"%R2_{region}" if "rsa" in method else f"R_{region}"

        cnn_scores = df.loc[(df['architecture'] == 'CNN') & (df['dataset'] == 'ImageNet1K'), score]
        vit_scores = df.loc[(df['architecture'] == 'VIT') & (df['dataset'] == 'ImageNet1K'), score]

        # Welch’s t-test
        t_stat, p_val = ttest_ind(cnn_scores, vit_scores, equal_var=False)
        print(f"[R_{region}] T-statistic: {t_stat:.3f}, p-value: {p_val:.3g}")
        print(f"[R_{region}] Mean CNN: {cnn_scores.mean():.3f}, Mean Transformer: {vit_scores.mean():.3f}, Diff: {(cnn_scores.mean()-vit_scores.mean()):.3f}")

        # Jittered scatter
        x_cnn = np.random.normal(i - 0.15, 0.03, size=len(cnn_scores))
        x_vit = np.random.normal(i + 0.15, 0.03, size=len(vit_scores))

        ax.scatter(x_cnn, cnn_scores, color=cnn_color, alpha=0.7, edgecolor='k', linewidth=0.3)
        ax.scatter(x_vit, vit_scores, color=vit_color, alpha=0.7, edgecolor='k', linewidth=0.3)

        # Mean horizontal lines
        ax.plot([i - 0.3, i], [cnn_scores.mean(), cnn_scores.mean()], color=cnn_color, lw=2)
        ax.plot([i, i + 0.3], [vit_scores.mean(), vit_scores.mean()], color=vit_color, lw=2)

    # Beautify
    ax.set_xticks(range(1, len(regions) + 1))
    ax.set_xticklabels(regions, fontsize=15)
    ax.set_ylabel("Encoding Score (R)", fontsize=16)
    ax.set_title(f"CNN vs Transformer Encoding Scores for {method}", fontsize=17, weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='CNN', markerfacecolor=cnn_color, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Transformer', markerfacecolor=vit_color, markersize=10)
    ]
    ax.legend(handles=legend_elements, frameon=False, fontsize=14)

    plt.tight_layout()
    output_file = f"../plots/thesis/jitterplot/CNN_vs_VIT_{method}.png"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    plt.show()

if __name__ == '__main__':
    evaluations = [
        "encoding_imagenet", "rsa_imagenet",
        "encoding_natural", "rsa_natural",
        "encoding_synthetic", "rsa_synthetic",
        "encoding_illusion", "rsa_illusion"
    ]
    for method in evaluations:
        regions = ['V1', 'V2', 'V4', 'IT']
        jitter_plot(method,regions)
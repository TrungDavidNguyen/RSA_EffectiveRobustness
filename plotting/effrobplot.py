import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D


def performance_jitter_by_category(data_file, categories_file):
    # Load data
    df = pd.read_csv(data_file)
    categories_df = pd.read_csv(categories_file)

    # Keep only CNN architectures (as in original)
    categories_df = categories_df[categories_df["architecture"] == "CNN"]

    # Merge dataset info from categories.csv
    df = pd.merge(df, categories_df[['Model', 'dataset']], on='Model', how='inner')

    # Melt wide dataframe to long format
    df_long = df.melt(
        id_vars=["Model", "dataset"],  # Use 'dataset' from categories.csv
        value_vars=["imagenet-r", "imagenet-sketch", "imagenet-a", "imagenetv2-matched-frequency"],
        var_name="eval_dataset",
        value_name="performance"
    )

    eval_datasets = df_long['eval_dataset'].unique()
    dataset_categories = df_long['dataset'].unique()

    # Assign colors dynamically to each dataset category
    cmap = cm.get_cmap('tab10', len(dataset_categories))
    category_colors = {cat: cmap(i) for i, cat in enumerate(dataset_categories)}

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, eval_ds in enumerate(eval_datasets, start=1):
        n_categories = len(dataset_categories)
        offsets = np.linspace(-0.2, 0.2, n_categories)  # spread for jitter

        for cat, offset in zip(dataset_categories, offsets):
            scores = df_long.loc[(df_long['dataset'] == cat) & (df_long['eval_dataset'] == eval_ds), 'performance']

            # Jitter x positions
            x_jitter = np.random.normal(i + offset, 0.03, size=len(scores))
            ax.scatter(x_jitter, scores, color=category_colors[cat], alpha=0.6, edgecolor='k', linewidth=0.3)

            # Plot mean
            mean_val = scores.mean()
            ax.plot([i + offset - 0.05, i + offset + 0.05], [mean_val, mean_val], color=category_colors[cat], lw=2)
            print(f"{eval_ds} - {cat} mean: {mean_val:.3f}")

    # Beautify plot
    ax.set_xticks(range(1, len(eval_datasets) + 1))
    ax.set_xticklabels(eval_datasets, rotation=30, fontsize=12)
    ax.set_ylabel("Performance", fontsize=14)
    ax.set_xlabel("Evaluation Dataset", fontsize=14)
    ax.set_title("Model Performance by Evaluation Dataset & Dataset Category", fontsize=16, weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Custom legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=cat,
                              markerfacecolor=color, markersize=10)
                       for cat, color in category_colors.items()]
    ax.legend(handles=legend_elements, frameon=False, fontsize=12, title='Dataset Category', bbox_to_anchor=(1.05, 1),
              loc='upper left')

    plt.tight_layout()

    # Save plot
    output_file = "../plots/performance_jitter_by_category.png"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    plt.show()


if __name__ == '__main__':
    performance_jitter_by_category("../results/effective_robustness.csv", "../results/categories.csv")

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from scipy.stats import ttest_ind


def performance_jitter_by_category(data_file, categories_file):
    # Load data
    df = pd.read_csv(data_file)
    categories_df = pd.read_csv(categories_file)

    categories_df = categories_df[categories_df["dataset"] != "JFT-300m"]

    # Merge dataset info from categories.csv
    df = pd.merge(df, categories_df[['Model', 'dataset']], on='Model', how='inner')

    # ✅ Rename column before melting
    df = df.rename(columns={"imagenetv2-matched-frequency": "imagenetv2"})

    # Melt wide dataframe to long format
    df_long = df.melt(
        id_vars=["Model", "dataset"],  # Use 'dataset' from categories.csv
        value_vars=["imagenet-r", "imagenet-sketch", "imagenet-a", "imagenetv2"],
        var_name="eval_dataset",
        value_name="performance"
    )

    eval_datasets = df_long['eval_dataset'].unique()
    dataset_categories = df_long['dataset'].unique()

    # Assign colors dynamically to each dataset category
    cmap = cm.get_cmap('viridis', len(dataset_categories))
    category_colors = {cat: cmap(i) for i, cat in enumerate(dataset_categories)}

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, eval_ds in enumerate(eval_datasets, start=1):
        n_categories = len(dataset_categories)
        offsets = np.linspace(-0.2, 0.2, n_categories)  # spread for jitter

        # Store mean scores per category
        cat_means = {}

        for cat, offset in zip(dataset_categories, offsets):
            scores = df_long.loc[
                (df_long['dataset'] == cat) & (df_long['eval_dataset'] == eval_ds),
                'performance'
            ]

            cat_means[cat] = scores.mean()

            # Jitter x positions
            x_jitter = np.random.normal(i + offset, 0.03, size=len(scores))
            ax.scatter(x_jitter, scores, color=category_colors[cat], alpha=0.6,
                       edgecolor='k', linewidth=0.3)

            # Plot mean
            ax.plot([i + offset - 0.05, i + offset + 0.05],
                    [scores.mean(), scores.mean()],
                    color=category_colors[cat], lw=2)
            print(f"{eval_ds} - {cat} mean: {scores.mean():.3f}")

        # ✅ Perform pairwise Welch’s t-tests between dataset categories
        for j in range(len(dataset_categories)):
            for k in range(j + 1, len(dataset_categories)):
                cat1, cat2 = dataset_categories[j], dataset_categories[k]
                scores1 = df_long.loc[
                    (df_long['dataset'] == cat1) & (df_long['eval_dataset'] == eval_ds),
                    'performance'
                ]
                scores2 = df_long.loc[
                    (df_long['dataset'] == cat2) & (df_long['eval_dataset'] == eval_ds),
                    'performance'
                ]
                if len(scores1) > 1 and len(scores2) > 1:  # avoid empty or single-value
                    t_stat, p_val = ttest_ind(scores1, scores2, equal_var=False)
                    diff = scores1.mean() - scores2.mean()
                    print(f"[{eval_ds}] {cat1} vs {cat2}: "
                          f"T={t_stat:.3f}, p={p_val:.3g}, "
                          f"Mean1={scores1.mean():.3f}, Mean2={scores2.mean():.3f}, Diff={diff:.3f}")

    # Beautify plot
    ax.set_xticks(range(1, len(eval_datasets) + 1))
    ax.set_xticklabels(eval_datasets, rotation=30, fontsize=12)
    ax.set_ylabel("Performance", fontsize=14)
    ax.set_xlabel("Evaluation Dataset", fontsize=14)
    ax.set_title("Model Performance by Evaluation Dataset & Dataset Category",
                 fontsize=16, weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=cat,
               markerfacecolor=color, markersize=10)
        for cat, color in category_colors.items()
    ]
    ax.legend(handles=legend_elements, frameon=False, fontsize=12,
              title='Dataset Category', bbox_to_anchor=(1.05, 1),
              loc='upper left')

    plt.tight_layout()

    # Save plot
    output_file = "../plots/performance_jitter_by_category.png"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    plt.show()


if __name__ == '__main__':
    performance_jitter_by_category("../results/effective_robustness.csv", "../results/categories.csv")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import matplotlib.cm as cm


def plot_brain_region_bars(dataset, eval_type="rsa", all_models=False):
    roi_names = ["V1", "V2", "V4", "IT"]
    evaluations = [
        f"{eval_type}_natural",
        f"{eval_type}_imagenet",
        f"{eval_type}_synthetic",
        f"{eval_type}_illusion"
    ]

    robustness_df = pd.read_csv("../results/effective_robustness.csv")
    categories_df = pd.read_csv("../results/categories.csv")

    r_values = []
    p_values = []

    for evaluation in evaluations:
        roi_prefix = "%R2_" if "rsa" in evaluation else "R_"
        brain_similarity_df = pd.read_csv(f"../results/{evaluation}.csv").dropna(
            subset=[roi_prefix + roi for roi in roi_names])

        df = pd.merge(brain_similarity_df, robustness_df, on='Model', how='inner')
        df = pd.merge(df, categories_df, on='Model', how='inner')

        if dataset == "imagenet-a":
            df = df[df['Model'].str.lower() != "resnet50"]

        if not all_models:
            df = df[df["architecture"] == "CNN"]

        roi_corrs = []
        roi_ps = []
        for roi in roi_names:
            result = linregress(df[roi_prefix + roi], df[dataset])
            roi_corrs.append(result.rvalue)
            roi_ps.append(result.pvalue)
        r_values.append(roi_corrs)
        p_values.append(roi_ps)

    r_df = pd.DataFrame(r_values, index=evaluations, columns=roi_names)
    p_df = pd.DataFrame(p_values, index=evaluations, columns=roi_names)

    # Plotting
    x = np.arange(len(roi_names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = cm.get_cmap("viridis", len(evaluations))
    colors = [cmap(i) for i in range(len(evaluations))]
    for i, evaluation in enumerate(evaluations):
        bars = ax.bar(x + i * width, r_df.loc[evaluation], width, label=evaluation, color=colors[i % len(colors)])
        # Annotate r and p values
        for j, bar in enumerate(bars):
            p_val = p_df.loc[evaluation, roi_names[j]]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"p={p_val:.3f}",
                ha='center',
                va='bottom',
                fontsize=9,
                color='black'
            )

    ax.set_xticks(x + width * (len(evaluations) / 2 - 0.5))
    ax.set_xticklabels(roi_names)
    ax.set_xlabel("Brain Regions")
    ax.set_ylabel("Correlation with " + dataset)
    ax.set_title(f"{eval_type.upper()} correlations for {dataset}")
    ax.legend(title="Evaluation", loc='lower left')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_brain_region_bars("imagenet-r", "rsa")
    #plot_brain_region_bars(evaluation, all_models=True)

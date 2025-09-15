import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import utils
from utils import PlottingConfig
def stimuli_comparison_plot(eval):
    evaluations = PlottingConfig.EVALUATIONS_DICT[eval]
    rois = PlottingConfig.ROIS
    legend_map = PlottingConfig.MAP_DATASET_NAMES_LONG

    categories_df = utils.load_categories_df()
    avg_scores = {region: [] for region in rois}

    for eval_name in evaluations:
        score_df = utils.load_eval_df(eval_name)
        merged_df = utils.merge_on_model(categories_df, score_df)

        for region in rois:
            score_col = utils.get_roi_col_name(region, eval_name)
            avg_score = merged_df[score_col].mean(skipna=True)
            avg_scores[region].append(avg_score)

    avg_scores_df = pd.DataFrame(avg_scores, index=evaluations).T

    x = np.arange(len(rois))
    width = 0.8 / len(evaluations)  # Dynamically scale bar width
    cmap = cm.get_cmap("viridis", len(evaluations))

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, eval_name in enumerate(evaluations):
        bars = ax.bar(
            x + i * width,
            avg_scores_df[eval_name],
            width,
            label=legend_map.get(eval_name, eval_name),
            color=cmap(i),
        )

        # Add value labels above bars for clarity
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    ax.set_ylabel(PlottingConfig.MAP_EVAL_SCORE_NAME[eval], fontsize=14)
    ax.set_xticks(x + width * (len(evaluations) - 1) / 2)
    ax.set_xticklabels(rois)
    ax.legend(loc="upper left")

    plt.tight_layout()

    # Use the centralized save_plot utility
    utils.save_plot(f"StimuliComparison_{eval}.png", "../plots/barplots/stimuli_comparison/")
    plt.show()


if __name__ == "__main__":
    stimuli_comparison_plot("rsa")

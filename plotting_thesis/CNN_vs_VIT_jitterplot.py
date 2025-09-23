import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from matplotlib import cm
import utils
from utils import PlottingConfig
from matplotlib.lines import Line2D


def CNN_VIT_jitter_plot(eval, rois):
    df_eval = utils.load_eval_df(eval)
    df_categories = utils.load_categories_df()
    df_merged = utils.merge_on_model(df_eval, df_categories)
    df_merged.dropna()
    # Get two colors from viridis colormap
    cmap = cm.get_cmap('viridis', 2)
    cnn_color = cmap(0)  # one end of the colormap
    vit_color = cmap(1)  # the other end

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, roi in enumerate(rois, start=1):
        roi_col_name = utils.get_roi_col_name(roi,eval)
        df_cnn = utils.filter_df_by_architecture(df_merged, "CNN")
        df_vit = utils.filter_df_by_architecture(df_merged, "VIT")

        cnn_scores = df_cnn[roi_col_name]
        vit_scores = df_vit[roi_col_name]

        # Welch’s t-test
        t_stat, p_val = ttest_ind(cnn_scores, vit_scores, equal_var=False)
        print(f"[R_{roi}] T-statistic: {t_stat:.3f}, p-value: {p_val:.3g}")
        print(f"[R_{roi}] Mean CNN: {cnn_scores.mean():.3f}, Mean Transformer: {vit_scores.mean():.3f}, Diff: {(cnn_scores.mean()-vit_scores.mean()):.3f}")

        # Jittered scatter
        x_cnn = np.random.normal(i - 0.15, 0.03, size=len(cnn_scores))
        x_vit = np.random.normal(i + 0.15, 0.03, size=len(vit_scores))

        ax.scatter(x_cnn, cnn_scores, color=cnn_color, alpha=0.7, edgecolor='k', linewidth=0.3)
        ax.scatter(x_vit, vit_scores, color=vit_color, alpha=0.7, edgecolor='k', linewidth=0.3)

        # Mean horizontal lines
        ax.plot([i - 0.3, i], [cnn_scores.mean(), cnn_scores.mean()], color=cnn_color, lw=2)
        ax.plot([i, i + 0.3], [vit_scores.mean(), vit_scores.mean()], color=vit_color, lw=2)
    eval_name = PlottingConfig.MAP_DATASET_TO_EVAL[eval]
    ax.set_xticks(range(1, len(rois) + 1))
    ax.set_xticklabels(rois, fontsize=15)
    ax.set_ylabel(PlottingConfig.MAP_EVAL_SCORE_NAME[eval_name], fontsize=16)
    ax.set_title(f"CNN vs Transformer {PlottingConfig.MAP_EVAL_CAPITALIZE[eval_name]} Scores for {PlottingConfig.MAP_DATASET_NAMES_LONG[eval]}", fontsize=17, weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='CNN', markerfacecolor=cnn_color, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Transformer', markerfacecolor=vit_color, markersize=10)
    ]
    ax.legend(handles=legend_elements, frameon=False, fontsize=14)

    plt.tight_layout()
    utils.save_plot(f"CNN_VIT_{eval}.png", "../plots/jitterplot/CNN_vs_VIT/")
    plt.show()


if __name__ == '__main__':
    evaluations = PlottingConfig.EVALUATIONS
    for eval in evaluations:
        rois = PlottingConfig.ROIS
        CNN_VIT_jitter_plot(eval, rois)
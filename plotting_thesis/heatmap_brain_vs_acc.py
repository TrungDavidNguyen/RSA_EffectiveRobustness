import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress

from utils import PlottingConfig
import utils


def create_heatmap(evaluation, all_models=False):
    roi_names = PlottingConfig.ROIS
    evaluations = PlottingConfig.EVALUATIONS_DICT[evaluation]

    # Load shared datasets
    accuracies_df = utils.load_accuracies_df()
    imagenet_acc = accuracies_df.columns[1]
    categories_df = utils.load_categories_df()

    eval_clean = [PlottingConfig.MAP_DATASET_NAMES_SHORT[e] for e in evaluations]

    r_value_matrix = pd.DataFrame(index=eval_clean, columns=roi_names)
    p_value_matrix = pd.DataFrame(index=eval_clean, columns=roi_names)

    for eval_name in evaluations:
        brain_similarity_df = utils.load_eval_df(eval_name).dropna(
            subset=[utils.get_roi_col_name(roi, eval_name) for roi in roi_names]
        )

        # Merge with model info
        df = brain_similarity_df.merge(accuracies_df, on='Model', how='inner')
        df = df.merge(categories_df, on='Model', how='inner')

        if not all_models:
            df = utils.filter_df_by_architecture(df, "CNN")

        for roi in roi_names:
            roi_col = utils.get_roi_col_name(roi, eval_name)
            result = linregress(df[roi_col], df[imagenet_acc])
            clean_label = PlottingConfig.MAP_DATASET_NAMES_SHORT[eval_name]
            r_value_matrix.loc[clean_label, roi] = result.rvalue
            p_value_matrix.loc[clean_label, roi] = result.pvalue

    r_value_matrix = r_value_matrix.astype(float)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(r_value_matrix, annot=True, cmap='coolwarm', vmin=-0.7, vmax=0.7,
                center=0, fmt=".2f", annot_kws={"size": 20})
    colorbar = plt.gcf().axes[-1]
    colorbar.tick_params(labelsize=16)

    for i in range(r_value_matrix.shape[0]):
        for j in range(r_value_matrix.shape[1]):
            p_val = p_value_matrix.iloc[i, j]
            text_color = 'black' if r_value_matrix.iloc[i, j] < 0.37 else 'white'
            plt.text(j + 0.5, i + 0.7, f"\n(p={p_val:.2f})",
                     ha='center', va='center', fontsize=16, color=text_color)

    model_type = "all models" if all_models else "only CNNs"
    plt.title(f"Correlation between {PlottingConfig.MAP_EVAL_CAPITALIZE[evaluation]} and ImageNet Accuracy", fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()

    output_path = os.path.join(PlottingConfig.PLOTS_DIR, "heatmap","brain_vs_acc", f"heatmap_{evaluation}.png")
    utils.save_plot(os.path.basename(output_path), os.path.dirname(output_path))
    plt.show()


if __name__ == '__main__':
    for evaluation in ["encoding", "rsa"]:
        create_heatmap(evaluation, all_models=False)

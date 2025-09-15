import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
import utils
from utils import PlottingConfig


def create_heatmap(evaluation, all_models=False):
    roi_names = PlottingConfig.ROIS

    robustness_df = utils.load_robustness_df()
    ood_datasets = robustness_df.columns[1:6]
    categories_df = utils.load_categories_df()
    brain_similarity_df = pd.read_csv(f"../results/{evaluation}.csv").dropna(
        subset=[utils.get_roi_col_name(roi, evaluation) for roi in roi_names])

    r_value_matrix = pd.DataFrame(index=ood_datasets, columns=roi_names)
    p_value_matrix = pd.DataFrame(index=ood_datasets, columns=roi_names)

    for ood_dataset in ood_datasets:
        for roi in roi_names:
            df = utils.merge_on_model(brain_similarity_df, robustness_df)
            df = utils.merge_on_model(df, categories_df)

            if ood_dataset == "imagenet-a":
                df = df[df['Model'].str.lower() != "resnet50"]

            if not all_models:
                df = utils.filter_df_by_architecture(df, "CNN")

            result = linregress(df[utils.get_roi_col_name(roi, evaluation)], df[ood_dataset])
            r_value_matrix.loc[ood_dataset, roi] = result.rvalue
            p_value_matrix.loc[ood_dataset, roi] = result.pvalue

    r_value_matrix = r_value_matrix.astype(float)
    r_value_matrix = r_value_matrix.rename(index={"imagenetv2-matched-frequency": "imagenetv2"})
    plt.figure(figsize=(10, 8))
    sns.heatmap(r_value_matrix, annot_kws={"size": 20}, annot=True, cmap='coolwarm', vmin=-0.7, vmax=0.7, center=0, fmt=".2f")
    colorbar = plt.gcf().axes[-1]  # get the last axis (the colorbar)
    colorbar.tick_params(labelsize=16)
    for i in range(len(p_value_matrix.columns)):
        for j in range(len(p_value_matrix.columns)):
            p = p_value_matrix.iloc[i, j]
            text_color = 'black' if r_value_matrix.iloc[i, j] < 0.37 else 'white'
            plt.text(j + 0.5, i + 0.7, f"\n(p={p:.2f})",
                     ha='center', va='center', fontsize=16, color=text_color)

    model_type = "all models" if all_models else "only CNNs"
    plt.title(f"{PlottingConfig.MAP_EVAL_CAPITALIZE[PlottingConfig.MAP_DATASET_TO_EVAL[evaluation]]} for {PlottingConfig.MAP_DATASET_NAMES_SHORT[evaluation]} vs Effective Robustness", fontsize=18)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.tight_layout()

    output_dir = f"../plots/heatmap/brain_vs_rob/{model_type}"
    utils.save_plot(f"heatmap_{evaluation}.png", output_dir)
    plt.show()
    #plt.close()


if __name__ == '__main__':
    evaluations = PlottingConfig.EVALUATIONS
    for evaluation in evaluations:
        create_heatmap(evaluation, True)

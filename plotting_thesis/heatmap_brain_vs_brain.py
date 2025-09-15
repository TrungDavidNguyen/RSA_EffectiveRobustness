import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from utils import PlottingConfig
import utils


def plot_heatmap(r_matrix, p_matrix, title, output_path):
    """
    Plot heatmap with r-values and p-values annotated.
    """
    plt.figure(figsize=(14, 9))
    sns.heatmap(r_matrix, annot=True, annot_kws={"size": 20}, cmap='coolwarm', vmin=-0.7, vmax=0.7, center=0, fmt=".2f")
    colorbar = plt.gcf().axes[-1]
    colorbar.tick_params(labelsize=16)

    for i in range(r_matrix.shape[0]):
        for j in range(r_matrix.shape[1]):
            p_val = p_matrix.iloc[i, j]
            text_color = 'black' if r_matrix.iloc[i, j] < 0.37 else 'white'
            plt.text(j + 0.5, i + 0.7, f"\n(p={p_val:.2f})",
                     ha='center', va='center', fontsize=16, color=text_color)

    plt.title(title, fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    utils.save_plot(os.path.basename(output_path), os.path.dirname(output_path))
    plt.show()


def load_and_merge_data(eval_id, eval_ood, roi, categories, all_models):
    """
    Load data for two evaluations and merge with categories.
    """
    roi_id_name = utils.get_roi_col_name(roi, eval_id)
    roi_ood_name = utils.get_roi_col_name(roi, eval_ood)

    df_id = utils.load_eval_df(eval_id)[["Model", roi_id_name]].dropna()
    df_ood = utils.load_eval_df(eval_ood)[["Model", roi_ood_name]].dropna()

    df_id = df_id.rename(columns={roi_id_name: f"{roi_id_name}_id"})
    df_ood = df_ood.rename(columns={roi_ood_name: f"{roi_ood_name}_ood"})

    df = df_id.merge(df_ood, on="Model", how="inner").merge(categories, on="Model", how="inner")

    if not all_models:
        df = utils.filter_df_by_architecture(df, "CNN")

    return df, f"{roi_id_name}_id", f"{roi_ood_name}_ood"


def compute_heatmap_for_pairs(evaluations, roi_names, categories, all_models=False):
    """
    Compute r- and p-matrices for pairs of evaluations.
    """
    index_labels = []
    for i, eval_x in enumerate(evaluations):
        for eval_y in evaluations[i + 1:]:
            label = f"{PlottingConfig.MAP_DATASET_NAMES_SHORT[eval_x]} vs {PlottingConfig.MAP_DATASET_NAMES_SHORT[eval_y]}"
            index_labels.append(label)

    r_matrix = pd.DataFrame(index=index_labels, columns=roi_names, dtype=float)
    p_matrix = pd.DataFrame(index=index_labels, columns=roi_names, dtype=float)

    label_idx = 0
    for i, eval_x in enumerate(evaluations):
        for eval_y in evaluations[i + 1:]:
            label = index_labels[label_idx]
            for roi in roi_names:
                df, x_col, y_col = load_and_merge_data(eval_x, eval_y, roi, categories, all_models)
                r, p = spearmanr(df[x_col], df[y_col])
                r_matrix.loc[label, roi] = r
                p_matrix.loc[label, roi] = p
            label_idx += 1

    return r_matrix, p_matrix


def create_heatmap_different_stimuli(evaluation_type, all_models=False):
    roi_names = PlottingConfig.ROIS
    categories = utils.load_categories_df()
    evaluations = PlottingConfig.EVALUATIONS_DICT[evaluation_type]

    r_matrix, p_matrix = compute_heatmap_for_pairs(evaluations, roi_names, categories, all_models)

    model_type = "all models" if all_models else "only CNNs"
    title = f"Correlation between {PlottingConfig.MAP_EVAL_CAPITALIZE[evaluation_type]} Scores"
    output_path = os.path.join(PlottingConfig.PLOTS_DIR, "heatmap","brain_vs_brain", f"heatmap_{evaluation_type}.png")
    plot_heatmap(r_matrix, p_matrix, title, output_path)


def create_heatmap_same_stimuli(all_models=False):
    roi_names = PlottingConfig.ROIS
    categories = utils.load_categories_df()
    eval_pairs = {
        "encoding_natural": "rsa_natural",
        "encoding_imagenet": "rsa_imagenet",
        "encoding_synthetic": "rsa_synthetic",
        "encoding_illusion": "rsa_illusion"
    }

    index_labels = [PlottingConfig.MAP_DATASET_NAMES_SHORT[k] for k in eval_pairs.keys()]
    r_matrix = pd.DataFrame(index=index_labels, columns=roi_names, dtype=float)
    p_matrix = pd.DataFrame(index=index_labels, columns=roi_names, dtype=float)

    for idx, (eval_id, eval_ood) in enumerate(eval_pairs.items()):
        label = index_labels[idx]
        for roi in roi_names:
            df, x_col, y_col = load_and_merge_data(eval_id, eval_ood, roi, categories, all_models)
            r, p = spearmanr(df[x_col], df[y_col])
            r_matrix.loc[label, roi] = r
            p_matrix.loc[label, roi] = p

    title = f"Correlation between RSA and Encoding Scores"
    output_path = os.path.join(PlottingConfig.PLOTS_DIR, "heatmap","brain_vs_brain", "heatmap_same_stimuli.png")
    plot_heatmap(r_matrix, p_matrix, title, output_path)


if __name__ == "__main__":
    for eval_type in ["encoding", "rsa"]:
        create_heatmap_different_stimuli(eval_type, all_models=True)

    create_heatmap_same_stimuli(all_models=True)

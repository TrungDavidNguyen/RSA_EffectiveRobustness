import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress


def create_heatmap(evaluation, all_models=False):
    """
    Create a heatmap showing correlation (r-value) between brain similarity scores and model accuracy.

    Parameters:
        evaluation (str): The evaluation type ('encoding' or 'rsa')
        all_models (bool): Whether to include all models or only CNNs trained on imagenet1k
    """
    roi_names = ["V1", "V2", "V4", "IT"]
    roi_prefix = "%R2_" if "rsa" in evaluation else "R_"
    evaluations = {
        "encoding": ["encoding_natural", "encoding_synthetic", "encoding_illusion"],
        "rsa": ["rsa_natural", "rsa_synthetic", "rsa_illusion"]
    }

    # Load shared datasets
    categories_df = pd.read_csv("../results/categories.csv")

    # Prepare index labels for rows in r_value_matrix
    index_labels = []
    for i, eval_x in enumerate(evaluations[evaluation]):
        for eval_y in evaluations[evaluation][i + 1:]:
            if evaluation == "encoding":
                label = f"{eval_x} vs {eval_y[9:]}"
            else:
                label = f"{eval_x} vs {eval_y}"
            index_labels.append(label)

    r_value_matrix = pd.DataFrame(index=index_labels, columns=roi_names)

    # Fill r_value_matrix with r-values
    label_idx = 0
    for i, eval_x in enumerate(evaluations[evaluation]):
        brain_similarity_df_x = pd.read_csv(f"../results/{eval_x}.csv").dropna(
            subset=[roi_prefix + roi for roi in roi_names])
        for eval_y in evaluations[evaluation][i + 1:]:
            brain_similarity_df_y = pd.read_csv(f"../results/{eval_y}.csv").dropna(
                subset=[roi_prefix + roi for roi in roi_names])
            df = pd.merge(brain_similarity_df_x, brain_similarity_df_y, on='Model', how='inner')
            df = pd.merge(df, categories_df, on='Model', how='inner')

            if not all_models:
                df = df[(df["dataset"] != "more data") & (df["architecture"] == "CNN")]

            for roi in roi_names:
                r_value = linregress(df[roi_prefix + roi + "_x"], df[roi_prefix + roi + "_y"]).rvalue
                r_value_matrix.loc[index_labels[label_idx], roi] = r_value
            label_idx += 1

    r_value_matrix = r_value_matrix.astype(float)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(r_value_matrix, annot=True, cmap='coolwarm', vmin=-0.7, vmax=0.7, center=0, fmt=".2f")
    model_type = "all_models" if all_models else "only_CNNs_imagenet1k"
    plt.title(f"Correlation between {evaluation} scores between datasets ({model_type})")
    plt.xlabel("ROI")
    plt.ylabel("fmri datasets")
    plt.tight_layout()

    # Save figure
    output_dir = f"../plots/heatmap_brain_vs_brain/{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/heatmap_{evaluation}.png")
    plt.close()


if __name__ == '__main__':
    for evaluation in ["encoding", "rsa"]:
        create_heatmap(evaluation)
        create_heatmap(evaluation, all_models=True)

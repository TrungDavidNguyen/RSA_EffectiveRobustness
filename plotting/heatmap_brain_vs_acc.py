import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress


def create_heatmap(evaluation, all_models = False):
    """
    Create a heatmap showing correlation (r-value) between brain similarity scores and model accuracy.

    Parameters:
        evaluation (str): The evaluation type (e.g., 'encoding', 'rsa', etc.)
        all_models (bool): Whether to include all models or only CNNs without 'more data'
    """
    roi_names = ["V1", "V2", "V4", "IT"]
    roi_prefix = "%R2_" if "rsa" in evaluation else "R_"

    # Load shared datasets
    accuracies_df = pd.read_csv("../results/accuracies.csv")
    ood_datasets = accuracies_df.columns[1:6]
    categories_df = pd.read_csv("../results/categories.csv")
    brain_similarity_df = pd.read_csv(f"../results/{evaluation}.csv").dropna(
        subset=[roi_prefix + roi for roi in roi_names])

    r_value_matrix = pd.DataFrame(index=ood_datasets, columns=roi_names)

    for ood_dataset in ood_datasets:
        for roi in roi_names:
            df = pd.merge(brain_similarity_df, accuracies_df, on='Model', how='inner')
            df = pd.merge(df, categories_df, on='Model', how='inner')

            if ood_dataset == "imagenet-a":
                df = df[df['Model'].str.lower() != "resnet50"]

            if not all_models:
                df = df[(df["dataset"] != "more data") & (df["architecture"] == "CNN")]

            # Linear regression between brain similarity and OOD accuracy
            r_value = linregress(df[roi_prefix + roi], df[ood_dataset]).rvalue
            r_value_matrix.loc[ood_dataset, roi] = r_value

    r_value_matrix = r_value_matrix.astype(float)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(r_value_matrix, annot=True, cmap='coolwarm', vmin=-0.6, vmax=0.6, center=0, fmt=".2f")
    model_type = "all_models" if all_models else "only_CNNs_imagenet1k"
    plt.title(f"Correlation between {evaluation} and accuracy ({model_type})")
    plt.xlabel("ROI")
    plt.ylabel("OOD Dataset")
    plt.tight_layout()

    # Save figure
    output_dir = f"../plots/heatmap_brain_vs_acc/{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/heatmap_{evaluation}.png")
    #plt.show()
    plt.close()


if __name__ == '__main__':
    evaluations = [
        "encoding", "rsa",
        "encoding_synthetic", "rsa_synthetic",
        "encoding_illusion", "rsa_illusion"
    ]
    for evaluation in evaluations:
        create_heatmap(evaluation)
        create_heatmap(evaluation, all_models=True)

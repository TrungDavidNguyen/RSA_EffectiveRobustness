import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress


def create_heatmap(evaluation, all_models = False):
    roi_names = ["V1", "V2", "V4", "IT"]
    roi_prefix = "%R2" if "rsa" in evaluation else "R"

    robustness_df = pd.read_csv("../results/effective_robustness.csv")
    ood_datasets = robustness_df.columns[1:6]
    categories_df = pd.read_csv("../results/categories.csv")
    brain_similarity_df = pd.read_csv(f"../results/effective_brain_similarity.csv")
    r_value_matrix = pd.DataFrame(index=ood_datasets, columns=roi_names)

    for ood_dataset in ood_datasets:
        for roi in roi_names:
            df = pd.merge(brain_similarity_df, robustness_df, on='Model', how='inner')
            df = pd.merge(df, categories_df, on='Model', how='inner')

            if ood_dataset == "imagenet-a":
                df = df[df['Model'].str.lower() != "resnet50"]

            if not all_models:
                df = df[(df["dataset"] != "more data") & (df["architecture"] == "CNN")]

            r_value = linregress(df[f"{roi_prefix}_{roi}_{evaluation}"], df[ood_dataset]).rvalue
            r_value_matrix.loc[ood_dataset, roi] = r_value

    r_value_matrix = r_value_matrix.astype(float)

    plt.figure(figsize=(10, 8))
    sns.heatmap(r_value_matrix, annot=True, cmap='coolwarm', vmin=-0.6, vmax=0.6, center=0, fmt=".2f")
    model_type = "all_models" if all_models else "only_CNNs_imagenet1k"
    plt.title(f"Correlation between effective brain similarity and effective robustness {evaluation} ({model_type})")
    plt.xlabel("ROI")
    plt.ylabel("OOD Dataset")
    plt.tight_layout()

    output_dir = f"../plots/heatmap_effbrain_vs_effrob/{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/heatmap_{evaluation}.png")
    #plt.show()
    plt.close()


if __name__ == '__main__':
    evaluations = [
        "encoding_synthetic", "rsa_synthetic",
        "encoding_illusion", "rsa_illusion"
    ]
    for evaluation in evaluations:
        create_heatmap(evaluation)
        create_heatmap(evaluation, all_models=True)

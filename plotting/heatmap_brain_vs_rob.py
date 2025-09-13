import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress


def create_heatmap(evaluation, all_models = False):
    roi_names = ["V1", "V2", "V4", "IT"]
    roi_prefix = "%R2_" if "rsa" in evaluation else "R_"

    robustness_df = pd.read_csv("../results/effective_robustness.csv")
    ood_datasets = robustness_df.columns[1:6]
    categories_df = pd.read_csv("../results/categories.csv")
    brain_similarity_df = pd.read_csv(f"../results/{evaluation}.csv").dropna(
        subset=[roi_prefix + roi for roi in roi_names])

    r_value_matrix = pd.DataFrame(index=ood_datasets, columns=roi_names)
    p_value_matrix = pd.DataFrame(index=ood_datasets, columns=roi_names)

    for ood_dataset in ood_datasets:
        for roi in roi_names:
            df = pd.merge(brain_similarity_df, robustness_df, on='Model', how='inner')
            df = pd.merge(df, categories_df, on='Model', how='inner')

            if ood_dataset == "imagenet-a":
                df = df[df['Model'].str.lower() != "resnet50"]

            if not all_models:
                df = df[df["architecture"] == "CNN"]
                df = df[df["dataset"] == "ImageNet1k"]


            result = linregress(df[roi_prefix + roi], df[ood_dataset])
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
    label = evaluation[9:] if "encoding" in evaluation else evaluation[4:]
    dataset_name = {
        "natural":"Natural",
        "illusion":"Illusion",
        "synthetic":"Synthetic",
        "imagenet":"ImageNet"
    }
    eval = "Encoding" if "encoding" in evaluation else "RSA"
    model_type = "all models" if all_models else "only CNNs"
    plt.title(f"{eval}–{dataset_name[label]} vs Effective Robustness", fontsize=18)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.tight_layout()

    output_dir = f"../plots/heatmap_brain_vs_rob/{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/heatmap_{evaluation}.png")
    plt.show()
    #plt.close()


if __name__ == '__main__':
    evaluations = [
        "encoding_natural", "rsa_natural",
        "encoding_synthetic", "rsa_synthetic",
        "encoding_illusion", "rsa_illusion",
        "encoding_imagenet", "rsa_imagenet"
    ]
    for evaluation in evaluations:
        create_heatmap(evaluation,True)

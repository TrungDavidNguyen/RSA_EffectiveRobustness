import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress


def create_heatmap(evaluation, all_models=False):
    roi_names = ["V1", "V2", "V4", "IT"]
    roi_prefix = "%R2_" if "rsa" in evaluation else "R_"
    evaluations = {
        "encoding": ["encoding_natural", "encoding_imagenet", "encoding_synthetic", "encoding_illusion"],
        "rsa": ["rsa_natural", "rsa_imagenet", "rsa_synthetic", "rsa_illusion"]
    }

    # Load shared datasets
    accuracies_df = pd.read_csv("../results/accuracies.csv")
    imagenet_acc = accuracies_df.columns[1]
    categories_df = pd.read_csv("../results/categories.csv")

    r_value_matrix = pd.DataFrame(index=evaluations[evaluation], columns=roi_names)
    p_value_matrix = pd.DataFrame(index=evaluations[evaluation], columns=roi_names)

    for eval in evaluations[evaluation]:
        brain_similarity_df = pd.read_csv(f"../results/{eval}.csv").dropna(
            subset=[roi_prefix + roi for roi in roi_names])

        # Merge with model info
        df = pd.merge(brain_similarity_df, accuracies_df, on='Model', how='inner')
        df = pd.merge(df, categories_df, on='Model', how='inner')

        if not all_models:
            df = df[df["architecture"] == "CNN"]

        for roi in roi_names:
            result = linregress(df[roi_prefix + roi], df[imagenet_acc])
            r_value_matrix.loc[eval, roi] = result.rvalue
            p_value_matrix.loc[eval, roi] = result.pvalue


    r_value_matrix = r_value_matrix.astype(float)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(r_value_matrix, annot=True, cmap='coolwarm', vmin=-0.7, vmax=0.7, center=0, fmt=".2f")
    for i in range(len(p_value_matrix.columns)):
        for j in range(len(p_value_matrix.columns)):
            p = p_value_matrix.iloc[i, j]
            text_color = 'black' if r_value_matrix.iloc[i, j] < 0.37 else 'white'
            plt.text(j + 0.5, i + 0.7, f"\n(p={p:.2f})",
                     ha='center', va='center', fontsize=10, color=text_color)

    model_type = "all models" if all_models else "only CNNs"
    plt.title(f"Correlation between {evaluation} and imagenet accuracy ({model_type})")
    plt.xlabel("ROI")
    plt.ylabel("fmri datasets")
    plt.tight_layout()

    # Save figure
    output_dir = f"../plots/heatmap_brain_vs_acc/{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/heatmap_{evaluation}.png")
    plt.show()
    #plt.close()


if __name__ == '__main__':
    for evaluation in ["encoding", "rsa"]:
        create_heatmap(evaluation)
        create_heatmap(evaluation, all_models=True)

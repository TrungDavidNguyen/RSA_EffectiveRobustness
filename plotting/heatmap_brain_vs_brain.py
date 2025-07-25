import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress


def create_heatmap_different_stimuli(evaluation, all_models=False):
    roi_names = ["V1", "V2", "V4", "IT"]
    roi_prefix = "%R2_" if "rsa" in evaluation else "R_"
    evaluations = {
        "encoding": ["encoding_natural", "encoding_synthetic", "encoding_illusion", "encoding_imagenet"],
        "rsa": ["rsa_natural", "rsa_synthetic", "rsa_illusion", "rsa_imagenet"]
    }

    categories_df = pd.read_csv("../results/categories.csv")

    index_labels = []
    for i, eval_x in enumerate(evaluations[evaluation]):
        for eval_y in evaluations[evaluation][i + 1:]:
            if evaluation == "encoding":
                label = f"{eval_x} vs {eval_y[9:]}"
            else:
                label = f"{eval_x} vs {eval_y}"
            index_labels.append(label)

    r_value_matrix = pd.DataFrame(index=index_labels, columns=roi_names)

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

    plt.figure(figsize=(10, 8))
    sns.heatmap(r_value_matrix, annot=True, cmap='coolwarm', vmin=-0.7, vmax=0.7, center=0, fmt=".2f")
    model_type = "all_models" if all_models else "only_CNNs_imagenet1k"
    plt.title(f"Correlation between {evaluation} scores between datasets ({model_type})")
    plt.xlabel("ROI")
    plt.ylabel("fmri datasets")
    plt.tight_layout()

    output_dir = f"../plots/heatmap_brain_vs_brain/{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/heatmap_{evaluation}.png")
    plt.close()


def create_heatmap_same_stimuli(all_models=False):
    roi_names = ["V1", "V2", "V4", "IT"]
    evaluations = {
        "encoding_natural": "rsa_natural",
        "encoding_synthetic": "rsa_synthetic",
        "encoding_illusion": "rsa_illusion",
        "encoding_imagenet": "rsa_imagenet"
    }

    categories_df = pd.read_csv("../results/categories.csv")

    index_labels = []
    for eval_x, eval_y in evaluations.items():
        label = f"{eval_x} vs {eval_y}"
        index_labels.append(label)

    r_value_matrix = pd.DataFrame(index=index_labels, columns=roi_names)

    label_idx = 0
    for eval_x, eval_y in evaluations.items():
        roi_prefix_x = "%R2_" if "rsa" in eval_x else "R_"
        brain_similarity_df_x = pd.read_csv(f"../results/{eval_x}.csv").dropna(
            subset=[roi_prefix_x + roi for roi in roi_names])
        roi_prefix_y = "%R2_" if "rsa" in eval_y else "R_"
        brain_similarity_df_y = pd.read_csv(f"../results/{eval_y}.csv").dropna(
            subset=[roi_prefix_y + roi for roi in roi_names])

        df = pd.merge(brain_similarity_df_x, brain_similarity_df_y, on='Model', how='inner')
        df = pd.merge(df, categories_df, on='Model', how='inner')

        if not all_models:
            df = df[(df["dataset"] != "more data") & (df["architecture"] == "CNN")]

        for roi in roi_names:
            r_value = linregress(df[roi_prefix_x + roi], df[roi_prefix_y + roi]).rvalue
            r_value_matrix.loc[index_labels[label_idx], roi] = r_value
        label_idx += 1

    r_value_matrix = r_value_matrix.astype(float)

    plt.figure(figsize=(10, 8))
    sns.heatmap(r_value_matrix, annot=True, cmap='coolwarm', vmin=-0.7, vmax=0.7, center=0, fmt=".2f")
    model_type = "all_models" if all_models else "only_CNNs_imagenet1k"
    plt.title(f"Correlation between scores between datasets ({model_type})")
    plt.xlabel("ROI")
    plt.ylabel("fmri datasets")
    plt.tight_layout()

    output_dir = f"../plots/heatmap_brain_vs_brain/{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/heatmap_same_stimuli.png")
    plt.close()

if __name__ == '__main__':
    for evaluation in ["encoding", "rsa"]:
        create_heatmap_different_stimuli(evaluation)
        create_heatmap_different_stimuli(evaluation, all_models=True)
    create_heatmap_same_stimuli()
    create_heatmap_same_stimuli(True)

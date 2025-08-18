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
    p_value_matrix = pd.DataFrame(index=index_labels, columns=roi_names)

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
                df = df[df["architecture"] == "CNN"]

            for roi in roi_names:
                result = linregress(df[roi_prefix + roi + "_x"], df[roi_prefix + roi + "_y"])
                r_value_matrix.loc[index_labels[label_idx], roi] = result.rvalue
                p_value_matrix.loc[index_labels[label_idx], roi] = result.pvalue

            label_idx += 1

    r_value_matrix = r_value_matrix.astype(float)
    p_value_matrix = p_value_matrix.astype(float)


    plt.figure(figsize=(10, 8))
    sns.heatmap(r_value_matrix, annot=True, cmap='coolwarm', vmin=-0.7, vmax=0.7, center=0, fmt=".2f")

    for i in range(len(p_value_matrix)):
        for j in range(len(p_value_matrix.columns)):
            p = p_value_matrix.iloc[i, j]
            text_color = 'black' if r_value_matrix.iloc[i, j] < 0.37 else 'white'
            plt.text(j + 0.5, i + 0.7, f"\n(p={p:.2f})",
                     ha='center', va='center', fontsize=10, color=text_color)

    model_type = "all models" if all_models else "only CNNs"
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
        "encoding_imagenet": "rsa_imagenet",
        "encoding_synthetic": "rsa_synthetic",
        "encoding_illusion": "rsa_illusion"
    }

    categories_df = pd.read_csv("../results/categories.csv")

    index_labels = []
    for eval_x, eval_y in evaluations.items():
        label = f"{eval_x} vs {eval_y}"
        index_labels.append(label)

    r_value_matrix = pd.DataFrame(index=index_labels, columns=roi_names)
    p_value_matrix = pd.DataFrame(index=index_labels, columns=roi_names)

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
            df = df[df["architecture"] == "CNN"]

        for roi in roi_names:
            result = linregress(df[roi_prefix_x + roi], df[roi_prefix_y + roi])
            r_value_matrix.loc[index_labels[label_idx], roi] = result.rvalue
            p_value_matrix.loc[index_labels[label_idx], roi] = result.pvalue

        label_idx += 1

    r_value_matrix = r_value_matrix.astype(float)
    p_value_matrix = p_value_matrix.astype(float)


    plt.figure(figsize=(10, 8))
    sns.heatmap(r_value_matrix, annot=True, cmap='coolwarm', vmin=-0.7, vmax=0.7, center=0, fmt=".2f")

    for i in range(len(p_value_matrix)):
        for j in range(len(p_value_matrix.columns)):
            p = p_value_matrix.iloc[i, j]
            text_color = 'black' if r_value_matrix.iloc[i, j] < 0.37 else 'white'
            plt.text(j + 0.5, i + 0.7, f"\n(p={p:.2f})",
                     ha='center', va='center', fontsize=10, color=text_color)

    model_type = "all models" if all_models else "only CNNs"
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

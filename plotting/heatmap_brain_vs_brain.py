import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def compute_r_p_matrices(df_x, df_y, roi_names, roi_prefix_x, roi_prefix_y):
    """
    Compute r-values and p-values matrices for given ROI dataframes using Spearman correlation.
    """
    r_matrix = pd.DataFrame(index=roi_names, columns=roi_names, dtype=float)
    p_matrix = pd.DataFrame(index=roi_names, columns=roi_names, dtype=float)

    for roi in roi_names:
        r, p = spearmanr(df_x[f"{roi_prefix_x}{roi}"], df_y[f"{roi_prefix_y}{roi}"])
        r_matrix.loc[roi, roi] = r
        p_matrix.loc[roi, roi] = p

    return r_matrix, p_matrix

def plot_heatmap(r_matrix, p_matrix, title, output_path):
    """
    Plot heatmap with r-values and p-values annotated.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(r_matrix, annot=True, cmap='coolwarm', vmin=-0.8, vmax=0.8, center=0, fmt=".2f")

    for i in range(r_matrix.shape[0]):
        for j in range(r_matrix.shape[1]):
            p_val = p_matrix.iloc[i, j]
            text_color = 'black' if r_matrix.iloc[i, j] < 0.37 else 'white'
            plt.text(j + 0.5, i + 0.7, f"\n(p={p_val:.2f})",
                     ha='center', va='center', fontsize=10, color=text_color)

    plt.title(title)
    plt.xlabel("ROI")
    plt.ylabel("fMRI datasets")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()


def load_and_merge_data(eval_id, eval_ood, roi, categories, all_models):
    """
    Load data for two evaluations and merge with categories.
    """
    roi_id_name = f"%R2_{roi}" if "rsa" in eval_id else f"R_{roi}"
    roi_ood_name = f"%R2_{roi}" if "rsa" in eval_ood else f"R_{roi}"

    df_id = pd.read_csv(f"../results/{eval_id}.csv")[[ "Model", roi_id_name]].dropna()
    df_ood = pd.read_csv(f"../results/{eval_ood}.csv")[[ "Model", roi_ood_name]].dropna()

    df_id = df_id.rename(columns={roi_id_name: f"{roi_id_name}_id"})
    df_ood = df_ood.rename(columns={roi_ood_name: f"{roi_ood_name}_ood"})

    df = df_id.merge(df_ood, on="Model", how="inner").merge(categories, on="Model", how="inner")

    if not all_models:
        df = df[df["architecture"] == "CNN"]
    else:
        df = df[df["architecture"] == "VIT"]

    return df, f"{roi_id_name}_id", f"{roi_ood_name}_ood"


def compute_heatmap_for_pairs(evaluations, roi_names, categories, all_models=False):
    """
    Compute r- and p-matrices for pairs of evaluations.
    """
    index_labels = []
    for i, eval_x in enumerate(evaluations):
        for eval_y in evaluations[i + 1:]:
            label = f"{eval_x[9:]} vs {eval_y[9:]}" if "encoding" in eval_x else f"{eval_x[4:]} vs {eval_y[4:]}"
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
    roi_names = ["V1", "V2", "V4", "IT"]
    categories = pd.read_csv("../results/categories.csv")
    evaluations_dict = {
        "encoding": ["encoding_natural", "encoding_synthetic", "encoding_illusion", "encoding_imagenet"],
        "rsa": ["rsa_natural", "rsa_synthetic", "rsa_illusion", "rsa_imagenet"]
    }
    evaluations = evaluations_dict[evaluation_type]

    r_matrix, p_matrix = compute_heatmap_for_pairs(evaluations, roi_names, categories, all_models)

    model_type = "all models" if all_models else "only CNNs"
    title = f"Correlation between {evaluation_type} scores ({model_type})"
    output_path = f"../plots/heatmap_brain_vs_brain/{model_type}/heatmap_{evaluation_type}.png"
    plot_heatmap(r_matrix, p_matrix, title, output_path)


def create_heatmap_same_stimuli(all_models=False):
    roi_names = ["V1", "V2", "V4", "IT"]
    categories = pd.read_csv("../results/categories.csv")
    evaluations = {
        "encoding_natural": "rsa_natural",
        "encoding_imagenet": "rsa_imagenet",
        "encoding_synthetic": "rsa_synthetic",
        "encoding_illusion": "rsa_illusion"
    }

    index_labels = [f"{e_id} vs {e_ood}" for e_id, e_ood in evaluations.items()]
    r_matrix = pd.DataFrame(index=index_labels, columns=roi_names, dtype=float)
    p_matrix = pd.DataFrame(index=index_labels, columns=roi_names, dtype=float)

    for idx, (eval_id, eval_ood) in enumerate(evaluations.items()):
        label = index_labels[idx]
        for roi in roi_names:
            df, x_col, y_col = load_and_merge_data(eval_id, eval_ood, roi, categories, all_models)
            r, p = spearmanr(df[x_col], df[y_col])
            r_matrix.loc[label, roi] = r
            p_matrix.loc[label, roi] = p

    model_type = "all models" if all_models else "only CNNs"
    title = f"Correlation between scores ({model_type})"
    output_path = f"../plots/heatmap_brain_vs_brain/{model_type}/heatmap_same_stimuli.png"
    plot_heatmap(r_matrix, p_matrix, title, output_path)


if __name__ == "__main__":
    for eval_type in ["encoding", "rsa"]:
        create_heatmap_different_stimuli(eval_type)
        create_heatmap_different_stimuli(eval_type, all_models=True)

    create_heatmap_same_stimuli()
    create_heatmap_same_stimuli(all_models=True)

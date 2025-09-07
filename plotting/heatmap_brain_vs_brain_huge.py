import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def create_heatmap_all_stimuli(evaluations_dict, all_models=False, output_name="heatmap_all_stimuli.png"):
    """
    Create a heatmap of Spearman correlations between model scores across ROIs and stimuli.
    Columns are renamed like 'natural_V1' instead of 'encoding_natural_R_V1' or 'rsa_natural_%R2_V1'.

    evaluations_dict: dict
        Keys are evaluation names (e.g., "encoding_natural" or "rsa_natural"), values are csv filenames (without .csv)
    all_models: bool
        Whether to include all models or only CNNs
    output_name: str
        Filename for saving the heatmap
    """
    roi_names = ["V1", "V2", "V4", "IT"]
    categories_df = pd.read_csv("../results/categories.csv")

    # Load all evaluation data
    eval_dfs = {}
    for eval_name, file_name in evaluations_dict.items():
        df = pd.read_csv(f"../results/{file_name}.csv")
        eval_dfs[eval_name] = df

    merged_df = None
    for eval_name, df in eval_dfs.items():
        roi_prefix = "%R2_" if "rsa" in eval_name else "R_"
        roi_cols = [roi_prefix + roi for roi in roi_names]
        stimulus = eval_name.split("_")[1]  # e.g., "natural" from "encoding_natural"
        df_renamed = df[["Model"] + roi_cols].copy()
        df_renamed = df_renamed.rename(columns={roi: f"{stimulus}_{roi}" for roi in roi_cols})
        if merged_df is None:
            merged_df = df_renamed
        else:
            merged_df = pd.merge(merged_df, df_renamed, on="Model", how="inner")

    # Filter CNNs if needed
    merged_df = pd.merge(merged_df, categories_df, on="Model", how="inner")
    if not all_models:
        merged_df = merged_df[merged_df["architecture"] == "CNN"]

    # Compute Spearman correlation across all columns
    score_cols = [col for col in merged_df.columns if col not in categories_df.columns and col != "Model"]
    corr_matrix = merged_df[score_cols].corr(method='spearman')

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, center=0)

    # Determine evaluation type for title
    eval_types = set([key.split("_")[0] for key in evaluations_dict.keys()])
    eval_type_str = " & ".join(eval_types).capitalize()

    model_type = "all models" if all_models else "only CNNs"
    plt.title(f"{eval_type_str}: Spearman correlation between model scores ({model_type})")
    plt.xlabel("Stimulus & ROI")
    plt.ylabel("Stimulus & ROI")
    plt.tight_layout()

    output_dir = f"../plots/heatmap_brain_vs_brain/{model_type}"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{output_name}")
    plt.show()


if __name__ == "__main__":
    evaluations_dict = {
        "encoding_natural": "encoding_natural",
        "encoding_synthetic": "encoding_synthetic",
        "encoding_illusion": "encoding_illusion",
        "encoding_imagenet": "encoding_imagenet"
    }

    #create_heatmap_all_stimuli(evaluations_dict, all_models=False, output_name="heatmap_encoding_CNN.png")
    create_heatmap_all_stimuli(evaluations_dict, all_models=False, output_name="heatmap_encoding_all_models.png")

    # Example for RSA
    rsa_dict = {
        "rsa_natural": "rsa_natural",
        "rsa_synthetic": "rsa_synthetic",
        "rsa_illusion": "rsa_illusion",
        "rsa_imagenet": "rsa_imagenet"
    }

    #create_heatmap_all_stimuli(rsa_dict, all_models=False, output_name="heatmap_rsa_CNN.png")
    create_heatmap_all_stimuli(rsa_dict, all_models=False, output_name="heatmap_rsa_all_models.png")

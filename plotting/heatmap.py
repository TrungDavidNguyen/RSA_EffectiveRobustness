import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import linregress


def create_heatmap(evaluation):
    roi_names = ["V1", "V2", "V4", "IT"]
    ood_datasets = pd.read_csv("../results/effective_robustness.csv").columns[1:6]
    roi_name = "%R2_" if "rsa" in evaluation else "R_"

    # Initialize a results DataFrame
    r_value_matrix = pd.DataFrame(index=ood_datasets, columns=roi_names)

    for ood_dataset in ood_datasets:
        for roi in roi_names:
            brain_similarity = pd.read_csv(f"../results/{evaluation}.csv")
            brain_similarity = brain_similarity.dropna(subset=[roi_name + roi])

            robustness = pd.read_csv("../results/effective_robustness.csv")
            df = pd.merge(brain_similarity, robustness, on='Model', how='inner')

            categories = pd.read_csv("../results/categories.csv")
            df = pd.merge(df, categories, on='Model', how='inner')
            if ood_dataset == "imagenet-a":
                df = df[df['Model'].str.lower() != "resnet50"]
                df = df.reset_index(drop=True)
            df = df[df["dataset"] != "more data"]
            df = df[df["architecture"] == "CNN"]
            # df = df[df["imagenet1k"] > 70]

            # Perform linear regression and store r-value
            slope, intercept, r_value, p_value, std_err = linregress(df[roi_name + roi], df[ood_dataset])
            r_value_matrix.loc[ood_dataset, roi] = r_value
    r_value_matrix = r_value_matrix.astype(float)
    print(r_value_matrix.columns)

    plt.figure(figsize=(10, 8))
    sns.heatmap(r_value_matrix, annot=True, cmap='coolwarm', vmin=-0.6, vmax=0.6, center=0, fmt=".2f")
    plt.title(f"Correlation between Brain Similarity ({evaluation}) and Effective Robustness")
    plt.xlabel("ROI")
    plt.ylabel("OOD Dataset")
    plt.tight_layout()
    plt.savefig(f"../plots/heatmap_{evaluation}")

    plt.show()


if __name__ == '__main__':
    create_heatmap("encoding")
    create_heatmap("rsa")
    create_heatmap("encoding_synthetic")
    create_heatmap("rsa_synthetic")
    create_heatmap("encoding_illusion")
    create_heatmap("rsa_illusion")

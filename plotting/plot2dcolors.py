import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


def create_plot(ood_dataset, roi_id, roi_ood, evaluation, evaluation_ood):
    """
    Creates a 2D scatter plot comparing brain similarity metrics for in-domain and out-of-domain datasets.

    Parameters:
        ood_dataset (str): Name of the out-of-domain dataset.
        roi_id (str): ROI for in-domain data.
        roi_ood (str): ROI for out-of-domain data.
        evaluation (str): Evaluation metric for in-domain data.
        evaluation_ood (str): Evaluation metric for out-of-domain data.
    """

    # Construct ROI column names
    roi_name_id = f"%R2_{roi_id}" if "rsa" in evaluation else f"R_{roi_id}"
    roi_name_ood = f"%R2_{roi_ood}" if "rsa" in evaluation_ood else f"R_{roi_ood}"

    # Load datasets
    brain_similarity = pd.read_csv(f"../results/{evaluation}.csv").dropna(subset=[roi_name_id])
    brain_similarity_ood = pd.read_csv(f"../results/{evaluation_ood}.csv").dropna(subset=[roi_name_ood])
    robustness = pd.read_csv("../results/effective_robustness.csv")
    categories = pd.read_csv("../results/categories.csv")

    # Keep only relevant columns and rename
    brain_similarity = brain_similarity[['Model', roi_name_id]].rename(
        columns={roi_name_id: f"{roi_name_id}_id"})
    brain_similarity_ood = brain_similarity_ood[['Model', roi_name_ood]].rename(
        columns={roi_name_ood: f"{roi_name_ood}_ood"})

    # Merge datasets
    df = brain_similarity.merge(robustness, on='Model', how='inner') \
        .merge(brain_similarity_ood, on='Model', how='inner') \
        .merge(categories, on='Model', how='inner')

    # Filter dataset based on OOD and architecture
    if ood_dataset.lower() == "imagenet-a":
        df = df[df['Model'].str.lower() != "resnet50"].reset_index(drop=True)
    df = df[df["architecture"] == "CNN"]

    # Linear regression
    x = df[f"{roi_name_id}_id"]
    y = df[f"{roi_name_ood}_ood"]
    slope, intercept, r_value, p_value, _ = linregress(x, y)
    y_fit = intercept + slope * x

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y_fit, color="red", label="Fit Line")

    # Display correlation stats
    plt.text(
        0.95, 0.95, f"r = {r_value:.2f}\np = {p_value:.2f}",
        transform=plt.gca().transAxes, ha='right', va='top',
        fontsize=12, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    # Scatter plot with color mapping
    vmin, vmax = np.percentile(df[ood_dataset].values, [5, 95])
    scatter = plt.scatter(
        x, y,
        c=df[ood_dataset].values,
        cmap='viridis',
        vmin=vmin, vmax=vmax,
        s=80, edgecolor='k'
    )
    plt.colorbar(scatter, label=f"{ood_dataset} Effective Robustness")

    # Labels and title
    plt.xlabel(f"{evaluation} {roi_name_id}")
    plt.ylabel(f"{evaluation_ood} {roi_name_ood}")
    plt.title(f"{evaluation} vs {evaluation_ood} for {roi_id} ROI")

    # Save figure
    os.makedirs("../plots/2d", exist_ok=True)
    plt.savefig(f"../plots/2d/{ood_dataset}_{evaluation}_{evaluation_ood}_{roi_name_id}_{roi_name_ood}.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    create_plot(
        ood_dataset="imagenet-r",
        roi_id="IT",
        roi_ood="IT",
        evaluation="rsa_synthetic",
        evaluation_ood="rsa_illusion"
    )

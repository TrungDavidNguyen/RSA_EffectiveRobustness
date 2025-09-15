import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from utils import PlottingConfig
sns.set(style="whitegrid")


def process_csv_files(folder_path, column_name):
    """Processes all CSV files in a folder to calculate the ratio of the max value index to the total number of rows."""
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    ratios = []

    for file in csv_files:
        df = pd.read_csv(file)
        if column_name in df.columns and not df.empty:
            max_index = df[column_name].idxmax()
            num_index = df.shape[0]
            if num_index > 0:
                ratios.append(max_index / num_index)

    return ratios


def plot_relative_depth_brain_regions(dataset):
    """Creates a 2x2 grid of histograms for the relative depth of brain regions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True, sharex=True)
    axes = axes.flatten()
    max_percent = 0

    rois = PlottingConfig.ROIS
    column_name = "R" if "encoding" in dataset else "%R2"
    folder_pattern = "../{dataset}/{brain_region}/encoding_{brain_region}_mean" if "encoding" in dataset \
        else "../{dataset}/{brain_region}/rsa_{brain_region}_mean"
    for i, region in enumerate(rois):
        ax = axes[i]
        folder_path = folder_pattern.format(dataset=dataset, brain_region=region)
        ratios = process_csv_files(folder_path, column_name)

        counts, bins = plt.np.histogram(ratios, bins=20, range=(0, 1), density=True)
        counts_percent = counts * 100 / counts.sum()

        # Plot the bar chart using calculated percentages
        ax.bar((bins[:-1] + bins[1:]) / 2, counts_percent, width=(bins[1] - bins[0]), color='skyblue',
               edgecolor='black')

        # Add a vertical line for the mean
        mean_ratio = sum(ratios) / len(ratios)
        ax.axvline(mean_ratio, color='red', linestyle='--', label=f'Mean = {mean_ratio:.2f}')

        ax.set_title(f"{region}", fontsize=20)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=18)

        # Keep track of the maximum percentage for setting the y-limit later
        if len(counts_percent) > 0:
            max_percent = max(max_percent, counts_percent.max())

    # Set common labels for the figure
    fig.supxlabel("Relative Depth", fontsize=20)
    fig.supylabel("Percentage of Models (%)", fontsize=20)
    dataset_name = PlottingConfig.MAP_DATASET_NAMES_LONG[dataset]
    eval = PlottingConfig.MAP_DATASET_TO_EVAL[dataset]
    eval_capitalize = PlottingConfig.MAP_EVAL_CAPITALIZE[eval]
    fig.suptitle(f"Relative Depth of best Layer for {eval_capitalize} with {dataset_name}", fontsize=24)
    # Apply the same y-axis limit to all subplots for consistent comparison
    for ax in axes:
        ax.set_ylim(0, 55)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle if needed

    utils.save_plot(f"{dataset}_2x2.png","../plots/histogram/avg_depth/")
    output_dir = f"../plots/depth_histogram/"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{dataset}_2x2.png")
    plt.show()


if __name__ == '__main__':
    datasets = PlottingConfig.EVALUATIONS_DICT["encoding"]
    for dataset in datasets:
        plot_relative_depth_brain_regions(dataset)

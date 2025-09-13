import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def process_csv_files(folder_path, column_name):
    """Processes all CSV files in a folder to calculate the ratio of the max value index to the total number of rows."""
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    ratios = []

    for file in csv_files:
        print(file)
        df = pd.read_csv(file)
        if column_name in df.columns and not df.empty:
            max_index = df[column_name].idxmax()
            num_index = df.shape[0]
            if num_index > 0:
                ratios.append(max_index / num_index)

    return ratios


def plot_relative_depth_brain_regions(brain_regions, dataset, column_name, folder_pattern):
    """Creates a 2x2 grid of histograms for the relative depth of brain regions."""
    # Create a 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True, sharex=True)
    # Flatten the 2D axes array to a 1D array for easy iteration
    axes = axes.flatten()

    max_percent = 0  # To synchronize the y-axis limit across all plots

    for i, region in enumerate(brain_regions):
        # Ensure we don't try to access an index out of bounds if more than 4 regions are provided
        if i >= 4:
            print(f"Warning: More than 4 brain regions provided. Only the first 4 will be plotted.")
            break

        ax = axes[i]
        folder_path = folder_pattern.format(dataset=dataset, brain_region=region)
        ratios = process_csv_files(folder_path, column_name)

        if not ratios:
            ax.set_title(f"{region}\n(No data)", fontsize=14)
            ax.text(0.5, 0.5, 'No data found', ha='center', va='center')
            continue

        # Create histogram with percentage on the y-axis
        counts, bins = [], []  # Default empty values
        # Use np.histogram to get counts and bins without plotting
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
    fig.supxlabel("Relative depth", fontsize=20)
    fig.supylabel("Percentage of models (%)", fontsize=20)
    label = dataset[9:] if "encoding" in dataset else dataset[4:]
    dataset_name = {
        "natural":"NSD Natural",
        "illusion":"Kamitani Illusion",
        "synthetic":"NSD Synthetic",
        "imagenet":"Kamitani ImageNet"
    }
    fig.suptitle(f"Relative Depth of layer best predicting {dataset_name[label]}", fontsize=24)
    # Apply the same y-axis limit to all subplots for consistent comparison
    for ax in axes:
        ax.set_ylim(0, 55)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle if needed

    output_dir = f"../plots/depth_histogram/"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{dataset}_2x2.png")
    plt.show()


if __name__ == '__main__':
    brain_regions = ['V1','V2', 'V4','IT',]  # example regions
    datasets = ["encoding_natural","encoding_imagenet", "encoding_synthetic", "encoding_illusion"]

    #datasets = ["rsa_natural","rsa_imagenet", "rsa_synthetic", "rsa_illusion"]
    #dataset = "encoding_synthetic"
    for dataset in datasets:
        plot_relative_depth_brain_regions(
            brain_regions,
            dataset,
            column_name='R',
            #column_name='%R2',
            folder_pattern="../{dataset}/{brain_region}/encoding_{brain_region}_subj1"
            #folder_pattern="../{dataset}/{brain_region}/rsa_{brain_region}_mean"
        )

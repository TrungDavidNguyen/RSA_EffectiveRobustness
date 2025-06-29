import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress


def create_bar_plot_by_model(method, rois, model=None):
    df = pd.read_csv(f"results/{method}.csv")
    eval = "%R2" if method in ["rsa", "rsa_synthetic"] else "R"

    # Set model as index
    df_plot = df.set_index('Model')
    order = [f"{eval}_{roi}" for roi in rois]
    df_plot = df_plot[order]

    if model is not None:
        df_plot = df_plot.loc[[model]]

    fig, ax = plt.subplots(figsize=(16, 6))  # Wider layout

    df_plot.plot(kind='bar', ax=ax)

    ax.set_xlabel('Model')
    ax.set_ylabel(method)
    ax.set_title(f'{method} {eval} results per Model')
    ax.legend(title=f'{method}/ROI')

    plt.tight_layout()
    plt.savefig(f"plots/barplot_{method}.png")
    plt.show()


def create_bar_plot_by_roi(method, roi):
    # Load data
    df = pd.read_csv(f"../results/{method}.csv")
    categories = pd.read_csv("../results/categories.csv")

    # Get architecture mapping per model
    architectures = categories["architecture"].unique()

    # Create color map
    colors = plt.cm.tab10.colors
    color_map = {arch: colors[i % len(colors)] for i, arch in enumerate(architectures)}

    # Determine which column to use for y-axis
    eval_col = "%R2" if method in ["rsa", "rsa_synthetic"] else "R"
    column_name = f"{eval_col}_{roi}"

    # Merge to get architecture info in df
    df = df.merge(categories[['Model', 'architecture']], left_on='Model', right_on='Model', how='left')

    # Assign colors to each bar based on architecture
    bar_colors = df['architecture'].map(color_map)

    # Create bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Model'], df[column_name], color=bar_colors)

    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel(f'{eval_col} value')
    plt.title(f'Method: {method} – ROI: {roi}')
    plt.xticks(rotation=45, ha='right')

    # Create legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[arch]) for arch in architectures]
    plt.legend(handles, architectures, title='Architecture')

    # Layout adjustment
    plt.tight_layout()
    os.makedirs("../plots/barplot", exist_ok=True)
    plt.savefig(f"../plots/barplot/barplot_{roi}_{method}.png")
    plt.show()


def corr(method, rois):
    nsd = pd.read_csv(f"../results/{method}.csv")
    nsd_synthetic = pd.read_csv(f"../results/{method}_synthetic.csv")

    df = pd.merge(nsd, nsd_synthetic, on='Model', how='inner', suffixes=(f' {method}', f' {method}_synthetic'))

    if method =="rsa":
        eval = "%R2"
    elif method == "encoding":
        eval = "R"
    df = df.dropna(subset=[f"{eval}_{rois[0]} {method}"])
    #df = df.head()
    slope, intercept, r_value, p_value, std_err = linregress(df[f"{eval}_{rois[0]} {method}_synthetic"], df[f"{eval}_{rois[0]} {method}"])
    print(r_value)

if __name__ == '__main__':
    #create_bar_plot_by_model("encoding_synthetic", ["V1","V2","V4","IT"])
    for roi in ["V1","V2","V4","IT"]:
        create_bar_plot_by_roi("rsa", roi)
        create_bar_plot_by_roi("encoding", roi)
        create_bar_plot_by_roi("rsa_synthetic", roi)
        create_bar_plot_by_roi("encoding_synthetic", roi)
        create_bar_plot_by_roi("rsa_illusion", roi)
        create_bar_plot_by_roi("encoding_illusion", roi)

import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import linregress
from effective_robustness import logit


def create_plot(roi, evaluation):
    if evaluation in ["rsa", "rsa_synthetic"]:
        eval_name = f"%R2_{evaluation}"
        roi_name = f"%R2_{roi}"
    elif evaluation in ["encoding", "encoding_synthetic"]:
        eval_name = f"R_{evaluation}"
        roi_name = f"R_{roi}"


    brain_similarity = pd.read_csv(f"results/{evaluation}.csv")
    brain_similarity = brain_similarity.dropna(subset=[roi_name])

    brain_similarity_synth = pd.read_csv(f"results/{evaluation}_synthetic.csv")

    cols_to_rename = {col: f"{col}_synthetic" for col in brain_similarity_synth.columns if
                      col != 'Model' and col in brain_similarity.columns}
    brain_similarity_synth_renamed = brain_similarity_synth.rename(columns=cols_to_rename)
    categories = pd.read_csv("results/categories.csv")

    df = pd.merge(brain_similarity, brain_similarity_synth_renamed, on='Model', how='inner')
    df = pd.merge(df, categories, on='Model', how='inner')
    df = df[df["architecture"] == "CNN"]
    if "encoding" in evaluation:
        df[roi_name] = logit(df[roi_name]*100)
        df[f"{roi_name}_synthetic"] = logit(df[f"{roi_name}_synthetic"]*100)
    else:
        df[roi_name] = logit(df[roi_name])
        df[f"{roi_name}_synthetic"] = logit(df[f"{roi_name}_synthetic"])

    # Define distinct markers for datasets
    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>']
    datasets = df["dataset"].unique()
    marker_map = {ds: markers[i % len(markers)] for i, ds in enumerate(datasets)}

    # Define a color map for architecture (optional)
    architectures = df["architecture"].unique()
    colors = plt.cm.tab10.colors
    color_map = {arch: colors[i % len(colors)] for i, arch in enumerate(architectures)}

    # Plot points with marker by dataset and color by architecture
    for _, row in df.iterrows():
        plt.scatter(row[roi_name], row[f"{roi_name}_synthetic"],
                    marker=marker_map[row["dataset"]],
                    color=color_map[row["architecture"]],
                    edgecolor='black',
                    s=50,
                    label=f'{row["dataset"]}_{row["architecture"]}')  # Temporary for deduplication

        plt.text(row[roi_name], row[f"{roi_name}_synthetic"], row["Model"],
                 fontsize=7, ha='right', va='bottom')

    # Regression line
    slope, intercept, r_value, p_value, std_err = linregress(df[roi_name], df[f"{roi_name}_synthetic"])
    print("slope", slope)
    print("intercept", intercept)
    x_vals = df[roi_name]
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color="red")

    # Correlation label
    plt.text(min(df[roi_name]), max(df[f"{roi_name}_synthetic"]), f"r = {r_value:.2f}")

    # Create custom legend entries for dataset (markers)
    from matplotlib.lines import Line2D
    dataset_handles = [Line2D([0], [0], marker=marker_map[ds], color='w', label=ds,
                              markerfacecolor='gray', markersize=8, markeredgecolor='black')
                       for ds in datasets]

    architecture_handles = [Line2D([0], [0], marker='o', color=color_map[arch], label=arch,
                                   linestyle='None', markersize=8)
                            for arch in architectures]

    legend1 = plt.legend(handles=dataset_handles, title="Dataset (Shape)", loc='lower left', fontsize=6, title_fontsize=8)
    plt.gca().add_artist(legend1)
    plt.legend(handles=architecture_handles, title="Architecture (Color)", loc='lower right', fontsize=6, title_fontsize=8)

    plt.xlabel(evaluation)
    plt.ylabel(f"{evaluation}_synthetic")
    plt.title(f"{evaluation}_synthetic vs {evaluation} {roi}")
    plt.tight_layout()
    os.makedirs("plots/brain_similarity", exist_ok=True)
    plt.savefig(f"plots/brain_similarity/{evaluation}_synthetic vs {evaluation}_{roi}")
    plt.show()


if __name__ == '__main__':
    for roi in ["V1", "V2", "V4", "IT"]:
        create_plot(roi, "rsa")
    for roi in ["V1", "V2", "V4", "IT"]:
        create_plot(roi, "encoding")

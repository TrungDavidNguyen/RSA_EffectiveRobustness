import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

def load_and_average_by_architecture_and_region(categories_path, evaluations, architectures, regions, results_dir):
    categories_df = pd.read_csv(categories_path)
    all_results = {}

    for eval_name in evaluations:
        score_df = pd.read_csv(os.path.join(results_dir, f"{eval_name}.csv"))
        merged_df = pd.merge(categories_df, score_df, on="Model", how="inner")
        #merged_df = merged_df[merged_df["dataset"] == "ImageNet1K"]

        data = [
            [
                merged_df.loc[merged_df["architecture"] == arch, f"R_{region}"].mean(skipna=True)
                for region in regions
            ]
            for arch in architectures
        ]
        df = pd.DataFrame(data, index=architectures, columns=regions)
        all_results[eval_name] = df

    return all_results

def plot_evaluation_overlap_per_region(all_avg_scores, regions, architectures, evaluations, title, output_file, legend_map=None):
    n_regions = len(regions)
    n_evals = len(evaluations)
    total_width = 0.8  # width of entire group per region
    eval_width = total_width / n_evals  # width of each evaluation bar
    cmap = cm.get_cmap("viridis", len(architectures))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_regions)

    # Draw bars
    for i, eval_name in enumerate(evaluations):
        offset = (i - (n_evals - 1)/2) * eval_width  # center the bars in the group

        for j, arch in enumerate(architectures):
            avg_scores_df = all_avg_scores[eval_name]
            values = avg_scores_df.loc[arch, regions].values

            ax.bar(
                x + offset,
                values,
                eval_width,
                label=f"{arch}" if i == 0 else None,  # show CNN/VIT legend only once
                color=cmap(j),
                alpha=0.7,
                edgecolor='black',
                zorder=j
            )

    ax.set_ylabel("Encoding R", fontsize=30)

    # Set x-ticks at the center of each group
    ax.set_xticks(x)
    ax.set_xticklabels([legend_map.get(r, r) if legend_map else r for r in regions], fontsize=24)

    # Add evaluation labels beneath each bar
    for i, eval_name in enumerate(evaluations):
        offset = (i - (n_evals - 1)/2) * eval_width
        for xi in range(n_regions):
            ax.text(
                x[xi] + offset,
                -0.05,  # slightly below zero
                eval_name.replace("encoding_", ""),  # clean up name
                ha='center',
                va='top',
                rotation=45,
                fontsize=20
            )

    ax.set_title(title,fontsize=26)
    ax.legend(loc="upper left", fontsize=20)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    plt.show()


def main():
    results_dir = "../results"

    evaluations = ["encoding_natural", "encoding_imagenet", "encoding_synthetic", "encoding_illusion"]
    #evaluations = ["rsa_natural", "rsa_imagenet", "rsa_synthetic", "rsa_illusion"]

    architectures = ["CNN", "VIT"]
    regions = ["V1", "V2", "V4", "IT"]
    legend_map = {"V1": "V1", "V2": "V2", "V4": "V4", "IT": "IT"}

    all_avg_scores = load_and_average_by_architecture_and_region(
        categories_path=os.path.join(results_dir, "categories.csv"),
        evaluations=evaluations,
        architectures=architectures,
        regions=regions,
        results_dir=results_dir,
    )

    output_file = "../plots/thesis/barplot/avgScores_overlap_per_eval.png"
    plot_evaluation_overlap_per_region(
        all_avg_scores,
        regions,
        architectures,
        evaluations,
        title="CNN vs VIT",
        output_file=output_file,
        legend_map=legend_map
    )



if __name__ == "__main__":
    main()

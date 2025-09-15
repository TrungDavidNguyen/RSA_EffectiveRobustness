import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import ttest_ind

def load_and_average_by_architecture_and_region(categories_path, evaluations, architectures, regions, results_dir):
    categories_df = pd.read_csv(categories_path)
    all_results = {}
    all_raw_scores = {}  # store raw scores for t-tests

    for eval_name in evaluations:
        score_df = pd.read_csv(os.path.join(results_dir, f"{eval_name}.csv"))
        merged_df = pd.merge(categories_df, score_df, on="Model", how="inner")
        merged_df = merged_df[merged_df["dataset"] == "ImageNet1K"]

        data = [
            [
                merged_df.loc[merged_df["architecture"] == arch, f"R_{region}"].mean(skipna=True)
                for region in regions
            ]
            for arch in architectures
        ]
        df = pd.DataFrame(data, index=architectures, columns=regions)
        all_results[eval_name] = df

        # store raw scores
        raw_scores = {arch: merged_df.loc[merged_df["architecture"] == arch, [f"R_{r}" for r in regions]] for arch in architectures}
        all_raw_scores[eval_name] = raw_scores

    return all_results, all_raw_scores

def perform_t_tests(all_raw_scores, architectures, regions, evaluations):
    ttest_results = {}

    for eval_name in evaluations:
        ttest_results[eval_name] = {}
        raw_scores = all_raw_scores[eval_name]

        for region in regions:
            scores1 = raw_scores[architectures[0]][f"R_{region}"].dropna()
            scores2 = raw_scores[architectures[1]][f"R_{region}"].dropna()

            t_stat, p_val = ttest_ind(scores1, scores2, equal_var=False)  # Welch's t-test
            ttest_results[eval_name][region] = {"t_stat": t_stat, "p_val": p_val}

    return ttest_results

def plot_evaluation_overlap_per_region(all_avg_scores, regions, architectures, evaluations, title, output_file, legend_map=None):
    n_regions = len(regions)
    n_evals = len(evaluations)
    total_width = 0.8
    eval_width = total_width / n_evals
    cmap = cm.get_cmap("viridis", len(architectures))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_regions)

    for i, eval_name in enumerate(evaluations):
        offset = (i - (n_evals - 1)/2) * eval_width
        for j, arch in enumerate(architectures):
            avg_scores_df = all_avg_scores[eval_name]
            values = avg_scores_df.loc[arch, regions].values
            ax.bar(
                x + offset,
                values,
                eval_width,
                label=f"{arch}" if i == 0 else None,
                color=cmap(j),
                alpha=0.7,
                edgecolor='black',
                zorder=j
            )

    ax.set_ylabel("Encoding R", fontsize=30)
    ax.set_xticks(x)
    ax.set_xticklabels([legend_map.get(r, r) if legend_map else r for r in regions], fontsize=24)

    for i, eval_name in enumerate(evaluations):
        offset = (i - (n_evals - 1)/2) * eval_width
        for xi in range(n_regions):
            ax.text(
                x[xi] + offset,
                -0.05,
                eval_name.replace("encoding_", ""),
                ha='center',
                va='top',
                rotation=45,
                fontsize=20
            )

    ax.set_title(title, fontsize=26)
    ax.legend(loc="upper left", fontsize=20)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    plt.show()

def main():
    results_dir = "../results"
    evaluations = ["encoding_natural", "encoding_imagenet", "encoding_synthetic", "encoding_illusion"]
    architectures = ["CNN", "VIT"]
    regions = ["V1", "V2", "V4", "IT"]
    legend_map = {"V1": "V1", "V2": "V2", "V4": "V4", "IT": "IT"}

    all_avg_scores, all_raw_scores = load_and_average_by_architecture_and_region(
        categories_path=os.path.join(results_dir, "categories.csv"),
        evaluations=evaluations,
        architectures=architectures,
        regions=regions,
        results_dir=results_dir,
    )

    # Perform t-tests
    ttest_results = perform_t_tests(all_raw_scores, architectures, regions, evaluations)
    for eval_name, region_results in ttest_results.items():
        print(f"\nT-tests for {eval_name}:")
        for region, res in region_results.items():
            print(f"  {region}: t = {res['t_stat']:.3f}, p = {res['p_val']:.4f}")

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

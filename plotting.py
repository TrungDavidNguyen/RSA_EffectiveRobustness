import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


def create_plot(ood_dataset, roi, evaluation):
    brain_similarity = pd.read_csv(f"results/{evaluation}.csv")
    robustness = pd.read_csv("results/effective_robustness.csv")
    df = pd.merge(brain_similarity, robustness, on='Model', how='inner')
    if ood_dataset == "imagenet-a":
        df = df[df['Model'].str.lower() != "resnet50"]
        df = df.reset_index()
    #df = df[~df['Model'].str.contains("Densenet", case=False, na=False)]
    #df = df.reset_index()
    df = df[~df['Model'].str.contains("squeezenet", case=False, na=False)]
    df = df.reset_index()
    # create scatter plot with model names
    if evaluation == "rsa":
        eval_name = f"%R2_{evaluation}"
        roi_name = f"%R2_{roi}"
    else:
        eval_name = f"R_{evaluation}"
        roi_name = f"R_{roi}"
    plt.scatter(df[roi_name], df[ood_dataset], marker="o", color="blue")

    for i in range(len(df)):
        plt.text(df.loc[i, roi_name], df.loc[i, ood_dataset], df.loc[i, "Model"],
                 fontsize=7, ha='right', va='bottom')

    # fit line
    slope, intercept, r_value, p_value, std_err = linregress(df[roi_name], df[ood_dataset])
    x_vals = df[roi_name]
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color="red")
    # add correlation
    plt.text(min(df[roi_name]), max(df[ood_dataset]), f"r = {r_value:.2f}")

    plt.xlabel(eval_name)
    plt.ylabel("Effective Robustness")
    plt.title(f"{roi} and {ood_dataset}")
    plt.savefig(f"plots/{roi}_{ood_dataset}_{evaluation}")
    plt.show()


if __name__ == '__main__':
    for ood_dataset in ["imagenet-r", "imagenet-sketch", "imagenetv2-matched-frequency", "imagenet-a"]:
        for roi in ["IT", "V4"]:
            create_plot(ood_dataset, roi, "encoding")
import pandas as pd
import numpy as np
from scipy.stats import linregress


def get_slope_intercept(id_dataset, ood_dataset):

    accuracies = pd.read_csv("results/accuracies.csv")
    categories = pd.read_csv("results/categories.csv")

    df = pd.merge(accuracies, categories, on='Model', how='inner')
    if ood_dataset == "imagenet-a_1":
        df = df[df["imagenet1k-subset-a"] < 91.86]
        ood_dataset = "imagenet-a"
    elif ood_dataset == "imagenet-a_2":
        df = df[df["imagenet1k-subset-a"] > 91.86]
        ood_dataset = "imagenet-a"

    df = df[df["architecture"] == "CNN"]
    df = df[df["dataset"] == "ImageNet1K"]
    df = df.reset_index(drop=True)

    result = linregress(logit(df[id_dataset]), logit(df[ood_dataset]))

    return result.slope, result.intercept


def logit(acc):
    return np.log(np.divide(acc / 100.0, 1.0 - acc / 100.0))


def inv_logit(acc):
    return (np.exp(acc)/(1 + np.exp(acc)))*100


def effective_robustness(id_accuracy, ood_accuracy, slope, intercept):
    y_pred_logit = logit(id_accuracy) * slope + intercept
    y_pred = inv_logit(y_pred_logit)
    eff_robust = ood_accuracy - y_pred
    return eff_robust


def effective_robustness_csv():
    datasets = {"imagenet-r": "imagenet1k-subset-r",
                "imagenet-sketch": "imagenet1k",
                "imagenetv2-matched-frequency": "imagenet1k",
                "imagenet-a_1": "imagenet1k-subset-a",
                "imagenet-a_2": "imagenet1k-subset-a"}
    acc = pd.read_csv("results/accuracies.csv")
    for col in acc.columns:
        if col == "imagenet-a":
            acc[col] = acc.apply(
                lambda row: effective_robustness(
                    row[datasets["imagenet-a_1"]],
                    row[col],
                    # resnet50 has 91.86 id accuracy
                    get_slope_intercept(datasets["imagenet-a_1"], "imagenet-a_1")[0] if row[datasets["imagenet-a_1"]] < 91.86 else  get_slope_intercept(datasets["imagenet-a_2"], "imagenet-a_2")[0],
                    get_slope_intercept(datasets["imagenet-a_1"], "imagenet-a_1")[1] if row[datasets["imagenet-a_1"]] < 91.86 else  get_slope_intercept(datasets["imagenet-a_2"], "imagenet-a_2")[1]
                ),
                axis=1
            )
        elif col not in ["imagenet1k", "Model", "imagenet1k-subset-r", "imagenet1k-subset-a"]:
            acc[col] = acc.apply(lambda row: effective_robustness(row[datasets[col]], row[col], get_slope_intercept(datasets[col], col)[0], get_slope_intercept(datasets[col], col)[1]), axis=1)
    acc.drop(columns=["imagenet1k","imagenet1k-subset-r", "imagenet1k-subset-a"], inplace=True)
    csv_filename = 'results/effective_robustness.csv'
    acc.to_csv(csv_filename, mode='w', index=False, header=True)


if __name__ == '__main__':
    effective_robustness_csv()

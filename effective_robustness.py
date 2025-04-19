import pandas as pd
import numpy as np


def logit(acc):
    return np.log(np.divide(acc / 100.0, 1.0 - acc / 100.0))


def inv_logit(acc):
    return (np.exp(acc)/(1 + np.exp(acc)))*100


def effective_robustness(id_accuracy, ood_accuracy, intercept, slope):
    y_pred_logit = logit(id_accuracy) * slope + intercept
    y_pred = inv_logit(y_pred_logit)
    eff_robust = ood_accuracy - y_pred
    return eff_robust


def generate_csv():
    # first element is intercept, second is slope, third is imagenet version
    line_fit = {"imagenet-r": [-2.1077156197680713,  0.636843984281646, "imagenet1k-subset-r"],
                "imagenet-sketch": [-2.370072912552283, 1.0709154135668684, "imagenet1k"],
                "imagenetv2-matched-frequency": [-0.4813069457734013, 0.9113725552359271, "imagenet1k"],
                "imagenet-a_1": [-5.14004617225444,   0.6894464990989362, "imagenet1k-subset-a"],
                "imagenet-a_2": [-10.27739708860645,  2.9559595169134285, "imagenet1k-subset-a"]}

    acc = pd.read_csv("results/accuracies.csv")
    for col in acc.columns:
        if col == "imagenet-a":
            acc[col] = acc.apply(
                lambda row: effective_robustness(
                    row[line_fit["imagenet-a_1"][2]],
                    row[col],
                    line_fit["imagenet-a_1"][0] if row["imagenet1k-subset-a"] < 91.86 else line_fit["imagenet-a_2"][0],
                    line_fit["imagenet-a_1"][1] if row["imagenet1k-subset-a"] < 91.86 else line_fit["imagenet-a_2"][1]
                ),
                axis=1
            )
        elif col not in ["imagenet1k", "Model", "imagenet1k-subset-r", "imagenet1k-subset-a"]:
            acc[col] = acc.apply(lambda row: effective_robustness(row[line_fit[col][2]], row[col], line_fit[col][0], line_fit[col][1]), axis=1)
    acc.drop(columns=["imagenet1k","imagenet1k-subset-r", "imagenet1k-subset-a"], inplace=True)
    csv_filename = 'results/effective_robustness.csv'
    acc.to_csv(csv_filename, mode='w', index=False, header=True)


if __name__ == '__main__':
    generate_csv()


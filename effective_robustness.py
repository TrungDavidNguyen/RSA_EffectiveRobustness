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
    # first element is intercept, second is slope
    line_fit = {"imagenet-r": [-1.5999151525728197,  0.9115905266235703], "imagenet-sketch": [-2.370072912552283, 1.0709154135668684], "imagenetv2-matched-frequency": [-0.4813069457734013, 0.9113725552359271], }

    acc = pd.read_csv("results/accuracies.csv")
    for col in acc.columns:
        if col == "imagenet-a":
            pass
        elif col not in ["imagenet1k", "Model"]:
            acc[col] = acc.apply(lambda row: effective_robustness(row['imagenet1k'], row[col], line_fit[col][0], line_fit[col][1]), axis=1)
    #TODO remove imagenet-a from cols and implement calculation for it
    acc.drop(columns=["imagenet1k", "imagenet-a"], inplace=True)
    csv_filename = 'results/effective_robustness.csv'
    acc.to_csv(csv_filename, mode='w', index=False, header=True)


if __name__ == '__main__':
    generate_csv()


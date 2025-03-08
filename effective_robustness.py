import torchvision.datasets as datasets
import torch
import numpy as np
from tqdm import tqdm


def logit(acc):
    return np.log(np.divide(acc / 100.0, 1.0 - acc / 100.0))


def inv_logit(acc):
    return (np.exp(acc)/(1 + np.exp(acc)))*100


def effective_robustness(id_accuracy, ood_accuracy, intercept, slope):
    y_pred_logit = logit(id_accuracy) * slope + intercept
    y_pred = inv_logit(y_pred_logit)
    eff_robust = ood_accuracy - y_pred
    return eff_robust


if __name__ == '__main__':
    # values for imagenet sketch
    intercept = -2.370072912552283
    slope = 1.0709154135668684
    id_accuracy = 76.13
    ood_accuracy = 24.0916519165039
    print("resnet",effective_robustness(intercept, slope))


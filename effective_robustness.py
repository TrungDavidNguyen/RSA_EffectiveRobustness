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


def measure_accuracy(model, path, transform, device="cuda" if torch.cuda.is_available() else "cpu"):
    dataset = datasets.ImageFolder(path, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.inference_mode():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) # Predictions for batch
            if isinstance(outputs, list): # if output is logits,apply softmax
                outputs = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item() # Add num of correct predictions
            total += labels.size(0)
    top1_accuracy = correct / total * 100
    return top1_accuracy


if __name__ == '__main__':
    print("resnet",effective_robustness(76.13,24.0916519165039,-2.370072912552283,1.0709154135668684))
    print("alexnet",effective_robustness(56.522003173828125,10.71940899,-2.370072912552283,1.0709154135668684))


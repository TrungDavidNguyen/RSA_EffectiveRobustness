import torchvision.datasets as datasets
import torch
import sys
from tqdm import tqdm
from torchvision import transforms as trn
from net2brain.architectures.pytorch_models import Standard


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
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item() # Add num of correct predictions
            total += labels.size(0)
    top1_accuracy = correct / total * 100
    return top1_accuracy


if __name__ == '__main__':
    num = int(sys.argv[1])
    models = {0: "Densenet161", 1: "Densenet169", 2: "Densenet201", 3: "GoogleNet"}
    #models = {0: "ResNet152", 1: "ResNet18", 2: "VGG13_bn", 3: "VGG16"}
    #models = {0: "VGG16_bn", 1: "VGG19", 2: "VGG19_bn", 3: "efficientnet_b1"}
    #models = {0: "efficientnet_b2", 1: "efficientnet_b3", 2: "efficientnet_b4", 3: "efficientnet_b5"}
    #models = {0: "efficientnet_b6", 1: "efficientnet_b7", 2: "mnasnet05", 3: "mnasnet10"}
    model_name = models[num]

    # get model
    standard = Standard(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    model = standard.get_model(pretrained=True)
    transform = trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("ImageNet Accuracy:", measure_accuracy(model, "/home/modelrep/public/dataset/imagenet", transform))
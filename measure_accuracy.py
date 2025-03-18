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
    models_list = [ 'Densenet161', 'Densenet169', 'Densenet201',
                   'GoogleNet', 'ResNet101', 'ResNet152', 'ResNet18', 'ResNet34',
                   'ShuffleNetV2x05', 'ShuffleNetV2x10', 'Squeezenet1_0', 'Squeezenet1_1',
                   'VGG11', 'VGG11_bn', 'VGG13', 'VGG13_bn', 'VGG16',
                   'VGG16_bn', 'VGG19', 'VGG19_bn', 'efficientnet_b0', 'efficientnet_b1',
                   'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                   'efficientnet_b6', 'efficientnet_b7', 'mnasnet05', 'mnasnet10', 'mobilenet_v2',
                   'mobilenet_v3_large', 'mobilenet_v3_small', 'mobilenet_v2','mobilenet_v3_large',
                   'mobilenet_v3_small','resnext101_32x8d', 'resnext50_32x4d', 'wide_resnet101_2', 'wide_resnet50_2']
    models = {}
    for i in range(len(models_list)):
        models[i] = models_list[i]
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

    print(model_name, " imagenet Accuracy:", measure_accuracy(model, "/scratch/modelrep/sadiya/students/david/RSA_EffectiveRobustness/imagenet", transform))
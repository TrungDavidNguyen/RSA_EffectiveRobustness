import torch
import os
import pandas as pd
import sys

from net2brain.architectures.pytorch_models import Standard
from torchvision.transforms import transforms as trn
from measure_accuracy import measure_accuracy_subset


def main(model_name):
    # measure accuracy
    # get model and transform
    models = Standard(model_name, "cuda" if torch.cuda.is_available() else "cpu")
    model = models.get_model(pretrained=True)
    transform = trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # get imagenet accuracy from csv
    df = pd.read_csv('results/accuracies.csv')
    id_accuracy = df.loc[df['Model'] == model_name, 'imagenet1k'].iloc[0]
    print(model_name, " id accuracy", id_accuracy)
    ood_path = "imagenet-val"
    ood_name = "imagenet-subset-a"
    ood_path_complete = os.path.join(os.getcwd(), ood_path)
    ood_accuracy = measure_accuracy_subset(model, ood_path_complete, transform)
    print(model_name, " ood accuracy", ood_accuracy)
    if ood_name not in df.columns:
        df[ood_name] = None
    df.loc[df['Model'] == model_name, ood_name] = ood_accuracy

    # Save to CSV
    csv_filename = f'results/accuracies.csv'
    df.to_csv(csv_filename, mode='w', index=False, header=True)


if __name__ == '__main__':
    num = int(sys.argv[1])
    models_list = ['ResNet50','AlexNet','Densenet121', 'Densenet161', 'Densenet169', 'Densenet201',
               'GoogleNet', 'ResNet101', 'ResNet152', 'ResNet18', 'ResNet34',
               'ShuffleNetV2x05', 'ShuffleNetV2x10', 'Squeezenet1_0', 'Squeezenet1_1',
               'VGG11', 'VGG11_bn', 'VGG13', 'VGG13_bn', 'VGG16',
               'VGG16_bn', 'VGG19', 'VGG19_bn', 'efficientnet_b0', 'efficientnet_b1',
               'mnasnet05', 'mnasnet10', 'mobilenet_v2',
               'mobilenet_v3_large', 'mobilenet_v3_small']
    model_name = models_list[num]
    main(model_name)

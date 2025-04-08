import torch
import os
import pandas as pd
import sys

from net2brain.architectures.pytorch_models import Standard
from torchvision.transforms import transforms as trn
from effective_robustness import measure_accuracy
from effective_robustness import effective_robustness


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
    df_imagenet = pd.read_csv('results/results-imagenet-sketch.csv')
    model_row = df_imagenet[df_imagenet['Model'] == model_name]
    id_accuracy = model_row['imagenet1k'].values[0]
    print(model_name," id accuracy", id_accuracy)
    ood_path = os.path.join(os.getcwd(), "imagenet-a")
    ood_accuracy = measure_accuracy(model, ood_path, transform)
    print(model_name," ood accuracy", ood_accuracy)
    # values for imagenet a
    intercept = -6.291670522669618
    slope = 3.051638725976987
    eff_robustness = effective_robustness(id_accuracy, ood_accuracy, intercept, slope)
    print(model_name, "eff robust", eff_robustness)
    df = pd.DataFrame(columns=['Model', 'eff.Robustness', 'imagenet1k', 'imagenet-a'])
    df.loc[len(df)] = [model_name,eff_robustness,id_accuracy,ood_accuracy]

    # Save to CSV
    csv_filename = 'results/results-imagenet-a.csv'
    file_exists = os.path.isfile(csv_filename)
    df.to_csv(csv_filename, mode='a', index=False, header=not file_exists)


if __name__ == '__main__':
    num =int(sys.argv[1])
    models_list = ['ResNet50','AlexNet','Densenet121', 'Densenet161', 'Densenet169', 'Densenet201',
               'GoogleNet', 'ResNet101', 'ResNet152', 'ResNet18', 'ResNet34',
               'ShuffleNetV2x05', 'ShuffleNetV2x10', 'Squeezenet1_0', 'Squeezenet1_1',
               'VGG11', 'VGG11_bn', 'VGG13', 'VGG13_bn', 'VGG16',
               'VGG16_bn', 'VGG19', 'VGG19_bn', 'efficientnet_b0', 'efficientnet_b1',
               'mnasnet05', 'mnasnet10', 'mobilenet_v2',
               'mobilenet_v3_large', 'mobilenet_v3_small']
    model_name = models_list[num]
    main(model_name)
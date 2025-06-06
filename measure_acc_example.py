import torch
import os
import pandas as pd
import sys

from net2brain.architectures.pytorch_models import Standard
from torchvision.transforms import transforms as trn
from measure_accuracy import measure_accuracy_subset_a
from measure_accuracy import measure_accuracy_subset_r
from measure_accuracy import measure_accuracy
from measure_accuracy import measure_accuracy_a
from measure_accuracy import measure_accuracy_r
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm import create_model


def get_standard_model(model_name):
    models = Standard(model_name, "cuda" if torch.cuda.is_available() else "cpu")
    model = models.get_model(pretrained=True)
    transform = trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return model, transform


def get_timm_model(model_name):
    model = create_model(model_name, pretrained=True)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return model, transform


def save_accuracy(model_name, dataset_name, accuracy):
    csv_filename = 'results/accuracies.csv'
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        if model_name in df['Model'].values:
            if dataset_name not in df.columns:
                df[dataset_name] = None
            df.loc[df['Model'] == model_name, dataset_name] = accuracy
        else:
            new_row = pd.Series({col: None for col in df.columns})
            new_row['Model'] = model_name
            new_row[dataset_name] = accuracy
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame({"Model": [model_name], dataset_name: [accuracy]})

    df.to_csv(csv_filename, index=False)


def timm(model_name, measure_acc_fn, dataset_name, dataset_path):
    # measure accuracy
    # get model and transform
    model, transform = get_timm_model(model_name)
    accuracy = measure_acc_fn(model, dataset_path, transform)
    print(model_name, " accuracy", accuracy)

    # Save to CSV
    save_accuracy(model_name, dataset_name, accuracy)


if __name__ == '__main__':
    num = int(sys.argv[1])
    """    models_list = ['ResNet50', 'AlexNet', 'Densenet121', 'Densenet161', 'Densenet169', 'Densenet201',
                       'GoogleNet', 'ResNet101', 'ResNet152', 'ResNet18', 'ResNet34',
                       'ShuffleNetV2x05', 'ShuffleNetV2x10', 'Squeezenet1_0', 'Squeezenet1_1',
                       'VGG11', 'VGG11_bn', 'VGG13', 'VGG13_bn', 'VGG16',
                       'VGG16_bn', 'VGG19', 'VGG19_bn', 'efficientnet_b0', 'efficientnet_b1',
                       'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                       'mnasnet05', 'mnasnet10', 'mobilenet_v2',
                       'mobilenet_v3_large', 'mobilenet_v3_small']"""
    """    models_list = ['inception_v3', 'inception_resnet_v2', 'xception',
                      'tf_efficientnet_b2_ns','tf_efficientnet_b4_ns',
                      'resnext50_32x4d', 'resnext101_32x8d',
                      'vit_base_patch16_224', 'vit_large_patch16_224',
                      'deit_base_patch16_224', 'swin_base_patch4_window7_224',
                      'mixer_b16_224', 'nfnet_l0', 'dm_nfnet_f0', 'regnety_032', 'regnety_080',
                      'coat_lite_mini','seresnet50',
                      'gluon_resnet50_v1c', 'gluon_resnext101_64x4d',
                      'wide_resnet50_2', 'convit_small']"""
    models_list = ['efficientnet_b3.ra2_in1k', 'beit_base_patch16_224.in22k_ft_in22k_in1k',
                   'gmlp_s16_224.ra3_in1k', 'convnext_base.fb_in22k_ft_in1k']
    model_name = models_list[num]
    dataset_path = os.path.join(os.getcwd(), "imagenet-val")
    timm(model_name, measure_accuracy, "imagenet1k", dataset_path)
    #dataset_path = os.path.join(os.getcwd(), "sketch")
    #timm(model_name, measure_accuracy, "imagenet-sketch", dataset_path)
    dataset_path = os.path.join(os.getcwd(), "imagenet-val")
    timm(model_name, measure_accuracy_subset_r, "imagenet1k-subset-r", dataset_path)
    dataset_path = os.path.join(os.getcwd(), "imagenet-val")
    timm(model_name, measure_accuracy_subset_a, "imagenet1k-subset-a", dataset_path)
    dataset_path = os.path.join(os.getcwd(), "imagenet-r")
    timm(model_name, measure_accuracy_r, "imagenet-r", dataset_path)
    #dataset_path = os.path.join(os.getcwd(), "imagenet-a")
    #timm(model_name, measure_accuracy_a, "imagenet-a", dataset_path)

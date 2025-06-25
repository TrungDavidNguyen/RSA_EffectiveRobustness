import os
import torch
import sys
import pandas as pd
import numpy as np
from utils.feature_extraction import FeatureExtractor
from utils.ridge_regression import RidgeCV_Encoding
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm import create_model
import torchextractor as tx


def save_encoding_results(features, fmri_dataset, roi_name, save_folder, num_subjects, model_name):
    current_dir = os.getcwd()

    R_sum = 0
    for subj in [1, 2, 3, 5, 6, 7]:
        roi_path = os.path.join(current_dir, fmri_dataset, roi_name, f"{roi_name}_fmri_subj{subj}")
        df = RidgeCV_Encoding(features, roi_path, model_name, np.logspace(-3, 3, 10),
                              save_path=f"{save_folder}/{roi_name}/encoding_{roi_name}_subj{subj}")
        df = df[['ROI', 'Layer', 'Model', 'R']]
        df = df.loc[[df['R'].idxmax()]]
        R_sum += df.loc[df.index[0], "R"]
        csv_filename = f'{save_folder}/{roi_name}/encoding_{roi_name}_subj{subj}/results-encoding-{roi_name}.csv'
        file_exists = os.path.isfile(csv_filename)
        df.to_csv(csv_filename, mode='a', index=False, header=not file_exists)
    R_mean = R_sum / num_subjects

    csv_filename = f'results/{save_folder}.csv'

    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        if model_name in df['Model'].values:
            if f"R_{roi_name}" not in df.columns:
                df[f"R_{roi_name}"] = None
            df.loc[df['Model'] == model_name, f"R_{roi_name}"] = R_mean
        else:
            new_row = pd.Series({col: None for col in df.columns})
            new_row['Model'] = model_name
            new_row[f"R_{roi_name}"] = R_mean
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame({"Model": [model_name], f"R_{roi_name}": [R_mean]})

    df.to_csv(csv_filename, index=False)


def encoding(model_name, netset, roi_name, stimuli_path, fmri_dataset, save_folder, num_subjects, features=None,
             device="cuda" if torch.cuda.is_available() else "cpu"):
    # Extract features
    if features is None:
        fx = FeatureExtractor(model=model_name, netset=netset, device=device)
        layers_to_extract = fx.get_all_layers()
        features = fx.extract(data_path=stimuli_path, consolidate_per_layer=False, layers_to_extract=layers_to_extract)
    save_encoding_results(features, fmri_dataset, roi_name, save_folder, num_subjects, model_name)

    return features


def encoding_custom(model_name, roi_name, stimuli_path, fmri_dataset, save_folder, num_subjects, features=None,
                    device="cuda" if torch.cuda.is_available() else "cpu"):
    # Extract features
    if features is None:
        model = create_model(model_name, pretrained=True)
        fx = FeatureExtractor(model=model, device=device, preprocessor=my_preprocessor, feature_cleaner=my_cleaner,
                              extraction_function=my_extractor)
        layers_to_extract = fx.get_all_layers()
        features = fx.extract(data_path=stimuli_path, consolidate_per_layer=False, layers_to_extract=layers_to_extract)
    save_encoding_results(features, fmri_dataset, roi_name, save_folder, num_subjects, model_name)
    return features


def my_preprocessor(image, model_name, device):
    """
    Args:
        image (Union[Image.Image, List[Image.Image]]): A PIL Image or a list of PIL Images.
        model_name (str): The name of the model, used to determine specific preprocessing if necessary.
        device (str): The device to which the tensor should be transferred ('cuda' for GPU, 'cpu' for CPU).

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: The preprocessed image(s) as PyTorch tensor(s).
    """

    model = create_model(model_name, pretrained=True)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    img_tensor = transform(image).unsqueeze(0)
    if device == 'cuda':
        img_tensor = img_tensor.cuda()

    return img_tensor


def my_extractor(preprocessed_data, layers_to_extract, model):
    # Create a extractor instance
    extractor_model = tx.Extractor(model, layers_to_extract)

    # Extract actual features
    _, features = extractor_model(preprocessed_data)

    return features


def my_cleaner(features):
    return features


if __name__ == '__main__':
    #num = int(sys.argv[1])
    standard = ['ResNet50', 'AlexNet', 'Densenet121', 'Densenet161', 'Densenet169',
                'Densenet201', 'GoogleNet', 'ResNet101', 'ResNet152', 'ResNet18',
                'ResNet34', 'ShuffleNetV2x05', 'ShuffleNetV2x10', 'Squeezenet1_0', 'Squeezenet1_1',
                'VGG11', 'VGG11_bn', 'VGG13', 'VGG13_bn', 'VGG16',
                'VGG16_bn', 'VGG19', 'VGG19_bn', 'efficientnet_b0', 'efficientnet_b1',
                'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'mnasnet05',
                'mnasnet10', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small']
    timm = ['inception_v3', 'inception_resnet_v2', 'xception', 'tf_efficientnet_b2_ns', 'tf_efficientnet_b4_ns',
            'resnext50_32x4d', 'resnext101_32x8d', 'vit_base_patch16_224', 'vit_large_patch16_224',
            'deit_base_patch16_224',
            'swin_base_patch4_window7_224', 'mixer_b16_224', 'nfnet_l0', 'dm_nfnet_f0', 'regnety_032',
            'regnety_080', 'coat_lite_mini', 'seresnet50', 'gluon_resnet50_v1c', 'gluon_resnext101_64x4d',
            'wide_resnet50_2', 'convit_small']
    timm_custom = ['efficientnet_b3.ra2_in1k', 'beit_base_patch16_224.in22k_ft_in22k_in1k', 'gmlp_s16_224.ra3_in1k',
                   'convnext_base.fb_in22k_ft_in1k']
    cornet = ["cornet_s", "cornet_z", "cornet_rt"]

    models_list = ['ResNet50', 'AlexNet', 'Densenet121', 'Densenet161', 'Densenet169',
                    'Densenet201', 'GoogleNet', 'ResNet101', 'ResNet152', 'ResNet18',
                    'ResNet34', 'ShuffleNetV2x05', 'ShuffleNetV2x10', 'Squeezenet1_0', 'Squeezenet1_1',
                    'VGG11', 'VGG11_bn', 'VGG13', 'VGG13_bn', 'VGG16',
                    'VGG16_bn', 'VGG19', 'VGG19_bn', 'efficientnet_b0', 'efficientnet_b1',
                    'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'mnasnet05',
                    'mnasnet10', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
                    'inception_v3', 'inception_resnet_v2', 'xception', 'tf_efficientnet_b2_ns', 'tf_efficientnet_b4_ns',
                    'resnext50_32x4d', 'resnext101_32x8d', 'vit_base_patch16_224', 'vit_large_patch16_224',
                    'deit_base_patch16_224',
                    'swin_base_patch4_window7_224', 'mixer_b16_224', 'nfnet_l0', 'dm_nfnet_f0', 'regnety_032',
                    'regnety_080', 'coat_lite_mini', 'seresnet50', 'gluon_resnet50_v1c', 'gluon_resnext101_64x4d',
                    'wide_resnet50_2', 'convit_small',
                    'efficientnet_b3.ra2_in1k', 'beit_base_patch16_224.in22k_ft_in22k_in1k', 'gmlp_s16_224.ra3_in1k',
                    'convnext_base.fb_in22k_ft_in1k',
                    "cornet_s", "cornet_z"]

    #model_name = models_list[num]
    for model_name in models_list:
        stimuli_path = os.path.join(os.getcwd(), "Illusion_Images")
        fmri_dataset = os.path.join("fmri", "fmri_illusion")
        save_folder = "encoding_illusion"
        if model_name in standard:
            encoding(model_name, "Standard", "V1", stimuli_path, fmri_dataset, save_folder, 6)
            encoding(model_name, "Standard", "V2", stimuli_path, fmri_dataset, save_folder, 6)
            encoding(model_name, "Standard", "V4", stimuli_path, fmri_dataset, save_folder, 6)
            encoding(model_name, "Standard", "IT", stimuli_path, fmri_dataset, save_folder, 6)
        elif model_name in timm:
            encoding(model_name, "Timm", "V1", stimuli_path, fmri_dataset, save_folder, 6)
            encoding(model_name, "Timm", "V2", stimuli_path, fmri_dataset, save_folder, 6)
            encoding(model_name, "Timm", "V4", stimuli_path, fmri_dataset, save_folder, 6)
            encoding(model_name, "Timm", "IT", stimuli_path, fmri_dataset, save_folder, 6)
        elif model_name in cornet:
            encoding(model_name, "Cornet", "V1", stimuli_path, fmri_dataset, save_folder, 6)
            encoding(model_name, "Cornet", "V2", stimuli_path, fmri_dataset, save_folder, 6)
            encoding(model_name, "Cornet", "V4", stimuli_path, fmri_dataset, save_folder, 6)
            encoding(model_name, "Cornet", "IT", stimuli_path, fmri_dataset, save_folder, 6)
        else:
            features = encoding_custom(model_name, "V1", stimuli_path, fmri_dataset, save_folder, 6)
            encoding_custom(model_name, "V2", stimuli_path, fmri_dataset, save_folder, 6, features)
            encoding_custom(model_name, "V4", stimuli_path, fmri_dataset, save_folder, 6, features)
            encoding_custom(model_name, "IT", stimuli_path, fmri_dataset, save_folder, 6, features)

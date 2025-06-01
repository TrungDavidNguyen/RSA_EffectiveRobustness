import os
import torch
import sys
import pandas as pd
import numpy as np
from utils.feature_extraction import FeatureExtractor
from utils.ridge_regression import RidgeCV_Encoding
from net2brain.utils.download_datasets import DatasetNSD_872
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm import create_model
import torchextractor as tx


def encoding(model_name, netset, roi_name, features=None):
    dataset = "encoding_synthetic"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    current_dir = os.getcwd()
    feat_path = f"{model_name}_Feat"
    #dataset_path = DatasetNSD_872.load_dataset()
    #stimuli_path = dataset_path["NSD_872_images"]
    stimuli_path = os.path.join(os.getcwd(),"NSD Synthetic", f"NSD_284_images")

    # Extract features
    if features is None:
        fx = FeatureExtractor(model=model_name, netset=netset, device=device)
        layers_to_extract = fx.get_all_layers()
        features = fx.extract(data_path=stimuli_path, save_path=feat_path, consolidate_per_layer=False,  layers_to_extract=layers_to_extract)
    R_sum = 0
    for subj in range(1, 9):
        roi_path = os.path.join(current_dir, f"fmri_synthetic/{roi_name}/{roi_name}_fmri_subj{subj}")
        df = RidgeCV_Encoding(features, roi_path, model_name, np.logspace(-3, 3, 10), save_path=f"{dataset}/{roi_name}/encoding_{roi_name}_subj{subj}")
        df = df[['ROI', 'Layer', 'Model', 'R']]
        df = df.loc[[df['R'].idxmax()]]
        R_sum += df.loc[df.index[0], "R"]
        csv_filename = f'{dataset}/{roi_name}/encoding_{roi_name}_subj{subj}/results-encoding-{roi_name}.csv'
        file_exists = os.path.isfile(csv_filename)
        df.to_csv(csv_filename, mode='a', index=False, header=not file_exists)
    R_mean = R_sum/8

    csv_filename = f'results/{dataset}.csv'

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
    return features

def encoding_custom(model_name, roi_name, features=None):
    dataset = "encoding_synthetic"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    current_dir = os.getcwd()
    feat_path = f"{model_name}_Feat"
    #dataset_path = DatasetNSD_872.load_dataset()
    #stimuli_path = dataset_path["NSD_872_images"]
    stimuli_path = os.path.join(os.getcwd(),"NSD Synthetic", f"NSD_284_images")

    # Extract features
    if features is None:
        model = create_model(model_name, pretrained=True)
        fx = FeatureExtractor(model=model, device=device, preprocessor=my_preprocessor, feature_cleaner=my_cleaner,
                              extraction_function=my_extractor)
        layers_to_extract = fx.get_all_layers()
        features = fx.extract(data_path=stimuli_path, save_path=feat_path, consolidate_per_layer=False,  layers_to_extract=layers_to_extract)
    R_sum = 0
    for subj in range(1, 9):
        roi_path = os.path.join(current_dir, f"fmri_synthetic/{roi_name}/{roi_name}_fmri_subj{subj}")
        df = RidgeCV_Encoding(features, roi_path, model_name, np.logspace(-3, 3, 10), save_path=f"{dataset}/{roi_name}/encoding_{roi_name}_subj{subj}")
        df = df[['ROI', 'Layer', 'Model', 'R']]
        df = df.loc[[df['R'].idxmax()]]
        R_sum += df.loc[df.index[0], "R"]
        csv_filename = f'{dataset}/{roi_name}/encoding_{roi_name}_subj{subj}/results-encoding-{roi_name}.csv'
        file_exists = os.path.isfile(csv_filename)
        df.to_csv(csv_filename, mode='a', index=False, header=not file_exists)
    R_mean = R_sum/8

    csv_filename = f'results/{dataset}.csv'

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
    num = int(sys.argv[1])
    models_list = ['vit_base_patch16_clip_224.openai', 'efficientnet_b3.ra2_in1k', 'vit_so400m_patch14_siglip_384', 'vit_base_patch16_224.dino', 'beit_base_patch16_224.in22k_ft_in22k_in1k',
                   'gmlp_s16_224.ra3_in1k', 'vit_base_patch16_224.mae', 'convnext_base.fb_in22k_ft_in1k']
    model_name = models_list[num]

    features = encoding_custom(model_name,  "V1")
    encoding_custom(model_name, "V2", features)
    encoding_custom(model_name, "V4", features)
    encoding_custom(model_name, "IT", features)
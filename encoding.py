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


def save_encoding_results(features, fmri_dataset, roi_name, save_folder, num_subjects):
    current_dir = os.getcwd()

    R_sum = 0
    for subj in range(1, num_subjects+1):
        roi_path = os.path.join(current_dir, f"{fmri_dataset}/{roi_name}/{roi_name}_fmri_subj{subj}")
        df = RidgeCV_Encoding(features, roi_path, model_name, np.logspace(-3, 3, 10), save_path=f"{save_folder}/{roi_name}/encoding_{roi_name}_subj{subj}")
        df = df[['ROI', 'Layer', 'Model', 'R']]
        df = df.loc[[df['R'].idxmax()]]
        R_sum += df.loc[df.index[0], "R"]
        csv_filename = f'{save_folder}/{roi_name}/encoding_{roi_name}_subj{subj}/results-encoding-{roi_name}.csv'
        file_exists = os.path.isfile(csv_filename)
        df.to_csv(csv_filename, mode='a', index=False, header=not file_exists)
    R_mean = R_sum/num_subjects

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


def encoding(model_name, netset, roi_name, stimuli_path, fmri_dataset, save_folder, num_subjects, features=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Extract features
    if features is None:
        fx = FeatureExtractor(model=model_name, netset=netset, device=device)
        layers_to_extract = fx.get_all_layers()
        features = fx.extract(data_path=stimuli_path, consolidate_per_layer=False,  layers_to_extract=layers_to_extract)
    save_encoding_results(features,fmri_dataset, roi_name, save_folder, num_subjects)

    return features


def encoding_custom(model_name, roi_name, stimuli_path, fmri_dataset, save_folder, num_subjects, features=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Extract features
    if features is None:
        model = create_model(model_name, pretrained=True)
        fx = FeatureExtractor(model=model, device=device, preprocessor=my_preprocessor, feature_cleaner=my_cleaner,
                              extraction_function=my_extractor)
        layers_to_extract = fx.get_all_layers()
        features = fx.extract(data_path=stimuli_path, consolidate_per_layer=False,  layers_to_extract=layers_to_extract)
    save_encoding_results(features, fmri_dataset, roi_name, save_folder, num_subjects)
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
    models_list = ['efficientnet_b3.ra2_in1k', 'beit_base_patch16_224.in22k_ft_in22k_in1k',
                   'gmlp_s16_224.ra3_in1k', 'convnext_base.fb_in22k_ft_in1k']
    model_name = models_list[num]

    stimuli_path = os.path.join(os.getcwd(), "NSD Dataset", "NSD_872_images")
    fmri_dataset = "fmri"
    save_folder = "encoding"

    features = encoding_custom(model_name,  "V1", stimuli_path, fmri_dataset, save_folder, 8)
    encoding_custom(model_name, "V2", stimuli_path, fmri_dataset, save_folder, 8, features)
    encoding_custom(model_name, "V4", stimuli_path, fmri_dataset, save_folder, 8, features)
    encoding_custom(model_name, "IT", stimuli_path, fmri_dataset, save_folder, 8, features)

import os
import torch
import sys
import pandas as pd
import numpy as np
from utils.feature_extraction import FeatureExtractor
from utils.ridge_regression import RidgeCV_Encoding
from net2brain.utils.download_datasets import DatasetNSD_872


def encoding(model_name, netset, roi_name):
    dataset = "encoding_synthetic"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    current_dir = os.getcwd()
    feat_path = f"{model_name}_Feat"
    dataset_path = DatasetNSD_872.load_dataset()
    stimuli_path = dataset_path["NSD_872_images"]

    # Extract features
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


if __name__ == '__main__':
    num = int(sys.argv[1])
    models_list = ['ResNet50', 'AlexNet', 'Densenet121', 'Densenet161', 'Densenet169', 'Densenet201',
                   'GoogleNet', 'ResNet101', 'ResNet152', 'ResNet18', 'ResNet34',
                   'ShuffleNetV2x05', 'ShuffleNetV2x10', 'Squeezenet1_0', 'Squeezenet1_1',
                   'VGG11', 'VGG11_bn', 'VGG13', 'VGG13_bn', 'VGG16',
                   'VGG16_bn', 'VGG19', 'VGG19_bn', 'efficientnet_b0', 'efficientnet_b1',
                   'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                   'mnasnet05', 'mnasnet10', 'mobilenet_v2',
                   'mobilenet_v3_large', 'mobilenet_v3_small']

    model_name = models_list[num]
    encoding(model_name, "Standard", "V4")
    encoding(model_name, "Standard", "IT")
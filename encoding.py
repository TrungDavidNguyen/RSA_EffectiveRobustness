import os
import torch
import sys
import pandas as pd
import numpy as np
from utils.feature_extraction import FeatureExtractor
from utils.ridge_regression import RidgeCV_Encoding
from net2brain.utils.download_datasets import DatasetNSD_872


def encoding(model_name, netset, roi_name, features=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    current_dir = os.getcwd()
    feat_path = f"{model_name}_Feat"
    dataset_path = DatasetNSD_872.load_dataset()
    stimuli_path = dataset_path["NSD_872_images"]

    # Extract features
    if features is None:
        fx = FeatureExtractor(model=model_name, netset=netset, device=device)
        layers_to_extract = fx.get_all_layers()
        features = fx.extract(data_path=stimuli_path, save_path=feat_path, consolidate_per_layer=False,  layers_to_extract=layers_to_extract)
    R_sum = 0
    for subj in range(1, 9):
        roi_path = os.path.join(current_dir, f"fmri/{roi_name}/{roi_name}_fmri_subj{subj}")
        df = RidgeCV_Encoding(features, roi_path, model_name, np.logspace(-3, 3, 10), save_path=f"encoding/{roi_name}/encoding_{roi_name}_subj{subj}")
        df = df[['ROI', 'Layer', 'Model', 'R']]
        df = df.loc[[df['R'].idxmax()]]
        R_sum += df.loc[df.index[0], "R"]
        csv_filename = f'encoding/{roi_name}/encoding_{roi_name}_subj{subj}/results-encoding-{roi_name}.csv'
        file_exists = os.path.isfile(csv_filename)
        df.to_csv(csv_filename, mode='a', index=False, header=not file_exists)
    R_mean = R_sum/8

    csv_filename = 'results/encoding.csv'

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

if __name__ == '__main__':
    num = int(sys.argv[1])

    standard = ["efficientnet_b5"]
    timm = ["convit_small", "inception_resnet_v2", "vit_large_patch16_224"]
    cornet = ["cornet_s", "cornet_z", "cornet_rt"]
    models_list = ['efficientnet_b1', "VGG19", "VGG19_bn", "cornet_s", "cornet_z"]
    model_name = models_list[num]
    if num > 2:
        features = encoding(model_name, "Cornet", "V1")
        encoding(model_name, "Cornet", "V2", features)
    else:
        features = encoding(model_name, "Standard", "V1")
        encoding(model_name, "Standard", "V2", features)

import os
import torch
import sys
import pandas as pd
import shutil
from net2brain.feature_extraction import FeatureExtractor
from net2brain.evaluations.encoding import Linear_Encoding
from net2brain.utils.download_datasets import DatasetNSD_872


def encoding(model_name, roi_name):
    netset = "Standard"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    current_dir = os.getcwd()
    feat_path = f"{model_name}_Feat"
    dataset_path = DatasetNSD_872.load_dataset()
    stimuli_path = dataset_path["NSD_872_images"]

    if not os.path.isdir(feat_path):
        # Extract features
        fx = FeatureExtractor(model=model_name, netset=netset, device=device)
        layers_to_extract = fx.get_all_layers()
        fx.extract(data_path=stimuli_path, save_path=feat_path, consolidate_per_layer=False,  layers_to_extract=layers_to_extract)
    R_sum = 0
    for subj in range(1,9):
        roi_path = os.path.join(current_dir, f"{roi_name}_fmri_subj{subj}")
        df = Linear_Encoding(feat_path, roi_path, model_name, save_path=f"Linear_Encoding_Results_{roi_name}_subj{subj}",)
        df = df[['ROI', 'Layer', 'Model', 'R']]
        df = df.loc[[df['R'].idxmax()]]
        R_sum += df.loc[df.index[0], "R"]
        csv_filename = f'Linear_Encoding_Results_{roi_name}_subj{subj}/results-encoding-{roi_name}.csv'
        file_exists = os.path.isfile(csv_filename)
        df.to_csv(csv_filename, mode='a', index=False, header=not file_exists)
    R_mean = R_sum/8
    df = pd.DataFrame({"ROI": [roi_name], "Model": [model_name], "R": [R_mean]
                       })
    csv_filename = f'results/results-encoding-{roi_name}.csv'

    file_exists = os.path.isfile(csv_filename)
    df.to_csv(csv_filename, mode='a', index=False, header=not file_exists)

    feat_path_complete = os.path.join(current_dir, f"{model_name}_Feat")
    if os.path.exists(feat_path_complete):
        shutil.rmtree(feat_path_complete)


if __name__ == '__main__':
    num = int(sys.argv[1])
    models_list = ['AlexNet','ResNet50','Densenet121', 'Densenet161', 'Densenet169', 'Densenet201',
               'GoogleNet', 'ResNet101', 'ResNet152', 'ResNet18', 'ResNet34',
               'ShuffleNetV2x05', 'ShuffleNetV2x10', 'Squeezenet1_0', 'Squeezenet1_1',
               'VGG11', 'VGG11_bn', 'VGG13', 'VGG13_bn', 'VGG16',
               'VGG16_bn', 'VGG19', 'VGG19_bn', 'efficientnet_b0', 'efficientnet_b1',
               'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
               'mnasnet05', 'mnasnet10', 'mobilenet_v2',
               'mobilenet_v3_large', 'mobilenet_v3_small','cornet_rt','cornet_s', 'cornet_z']
    model_name = models_list[num]
    encoding(model_name, "V4")
    encoding(model_name, "IT")


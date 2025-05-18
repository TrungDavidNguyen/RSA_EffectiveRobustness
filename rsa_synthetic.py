import os
import torch
import pandas as pd
import shutil
import sys
from net2brain.evaluations.rsa import RSA
from net2brain.utils.download_datasets import DatasetNSD_872
from net2brain.feature_extraction import FeatureExtractor
from net2brain.rdm_creation import RDMCreator
from utils.rsa_new import RSA


def main(model_name, netset, roi_name, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Set paths
    dataset = "NSD Synthetic"
    images = 0
    rdm_save_path = ""
    if dataset == "NSD Dataset":
        images = 872
        rdm_save_path = "rdm"
    elif dataset == "NSD Synthetic":
        images = 284
        rdm_save_path = "rdm_synthetic"


    current_dir = os.getcwd()
    save_path = os.path.join(current_dir, f"{model_name}_RDM")
    rdm_path = f"{model_name}_RDM"
    feat_path = f"{model_name}_Feat"
    # Load dataset
    #dataset_path = DatasetNSD_872.load_dataset()
    #stimuli_path = dataset_path["NSD_872_images"]
    stimuli_path = os.path.join(os.getcwd(),dataset, f"NSD_{images}_images")

    if not os.path.isdir(save_path):
        # Extract features
        fx = FeatureExtractor(model=model_name, netset=netset, device=device)
        layers_to_extract = fx.get_all_layers()
        fx.extract(data_path=stimuli_path, save_path=feat_path, consolidate_per_layer=True,  layers_to_extract=layers_to_extract)
        # Create RDM of model
        creator = RDMCreator(verbose=True, device=device)
        save_path = creator.create_rdms(feature_path=feat_path, save_path=rdm_path, save_format='npz')

    feat_path_complete = os.path.join(current_dir, feat_path)
    if os.path.exists(feat_path_complete):
        shutil.rmtree(feat_path_complete)

    brain_path = os.path.join(current_dir, rdm_save_path, roi_name)
    RSA(save_path, brain_path,model_name,roi_name, "rsa_synthetic")


if __name__ == '__main__':
    num = int(sys.argv[1])
    models_list = ['VGG19', 'VGG19_bn', 'efficientnet_b0']
    model_name = models_list[num]
    if num<=0:
        main(model_name, "Timm", "V1")
        main(model_name, "Timm", "V2")
    else:
        main(model_name, "Standard", "V1")
        main(model_name, "Standard", "V2")
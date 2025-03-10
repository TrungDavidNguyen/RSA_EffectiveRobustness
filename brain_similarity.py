import os
import sys
import torch
from net2brain.evaluations.rsa import RSA
from net2brain.utils.download_datasets import DatasetNSD_872
from net2brain.feature_extraction import FeatureExtractor
from net2brain.rdm_creation import RDMCreator
from net2brain.architectures.pytorch_models import Standard


def brain_similarity_rsa(model_name, netset, brain_path, roi, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    :param model_name: name of model from net2brain
    :param netset: netset of model from net2brain
    :param brain_path: path to brain RDMs
    :param roi: ROI we want to extract
    :param device: cpu or cuda
    :return: dataframe row with result
    """
    # Set paths
    current_dir = os.getcwd()
    save_path = os.path.join(current_dir, f"{model_name}_RDM")
    # Load dataset
    dataset_path = DatasetNSD_872.load_dataset()
    stimuli_path = dataset_path["NSD_872_images"]
    if not os.path.isdir(save_path):
        # Extract features
        feat_path = f"{model_name}_Feat"
        fx = FeatureExtractor(model=model_name, netset=netset, device=device)
        fx.extract(data_path=stimuli_path, save_path=feat_path, consolidate_per_layer=False)
        # Create RDM of model
        creator = RDMCreator(verbose=True, device=device)
        save_path = creator.create_rdms(feature_path=feat_path, save_path=f"{model_name}_RDM", save_format='npz')
    # Perform RSA
    return RSA_helper(save_path, brain_path, model_name, roi)


def brain_similarity_rsa_custom(model, model_name, my_preprocessor, my_cleaner, my_extractor, brain_path, roi, layers_to_extract, device):
    # Set paths
    current_dir = os.getcwd()
    save_path = os.path.join(current_dir, f"{model_name}_RDM")
    # Load dataset
    dataset_path = DatasetNSD_872.load_dataset()
    stimuli_path = dataset_path["NSD_872_images"]
    if not os.path.isdir(save_path):
        # Extract features
        feat_path = f"{model_name}_Feat"
        fx = FeatureExtractor(model=model, device=device, preprocessor=my_preprocessor, feature_cleaner=my_cleaner,
                              extraction_function=my_extractor)
        print(str(fx.get_all_layers()))
        fx.extract(data_path=stimuli_path, save_path=feat_path, consolidate_per_layer=False,  layers_to_extract=layers_to_extract)
        # Create RDM of model
        creator = RDMCreator(verbose=True, device=device)
        save_path = creator.create_rdms(feature_path=feat_path, save_path=f"{model_name}_RDM", save_format='npz')
    # Perform RSA
    return RSA_helper(save_path, brain_path, model_name, roi)


def RSA_helper(save_path, brain_path, model_name, roi):
    # picks row with highest %R2 of specified ROI
    evaluation = RSA(save_path, brain_path, model_name=model_name)
    df = evaluation.evaluate()
    df_roi = df[df["ROI"] == roi]
    max_r2_value = df_roi['%R2'].max()
    df_max_r2 = df_roi[df_roi['%R2'] == max_r2_value]
    df_filtered = df_max_r2[['ROI', 'Layer', 'Model', '%R2']]
    return df_filtered


if __name__ == '__main__':
    num = int(sys.argv[1])
    models = {0: "Densenet161", 1: "Densenet169", 2: "Densenet201", 3: "GoogleNet"}
    #models = {0: "ResNet152", 1: "ResNet18", 2: "VGG13_bn", 3: "VGG16"}
    #models = {0: "VGG16_bn", 1: "VGG19", 2: "VGG19_bn", 3: "efficientnet_b1"}
    #models = {0: "efficientnet_b2", 1: "efficientnet_b3", 2: "efficientnet_b4", 3: "efficientnet_b5"}
    #models = {0: "efficientnet_b6", 1: "efficientnet_b7", 2: "mnasnet05", 3: "mnasnet10"}

    model_name = models[num]
    # set path to the brain RDMs
    current_dir = os.getcwd()
    brain_path = os.path.join(current_dir, "NSD Dataset", "NSD_872_RDMs", "prf-visualrois", "combined")
    # get model
    standard = Standard(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    model = standard.get_model(pretrained=True)

    df = brain_similarity_rsa(model_name, "Standard", brain_path, "(3) V4_both_fmri")
    print(df)

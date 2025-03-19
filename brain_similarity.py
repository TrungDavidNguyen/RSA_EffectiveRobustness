import os
import torch
from net2brain.evaluations.rsa import RSA
from net2brain.utils.download_datasets import DatasetNSD_872
from net2brain.feature_extraction import FeatureExtractor
from net2brain.rdm_creation import RDMCreator


def brain_similarity_rsa(model_name, netset, brain_path, feat_path,rdm_path, roi, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    :param model_name: name of the model from net2brain
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
        fx = FeatureExtractor(model=model_name, netset=netset, device=device)
        layers_to_extract = fx.get_all_layers()
        fx.extract(data_path=stimuli_path, save_path=feat_path, consolidate_per_layer=False,  layers_to_extract=layers_to_extract)
        # Create RDM of model
        creator = RDMCreator(verbose=True, device=device)
        save_path = creator.create_rdms(feature_path=feat_path, save_path=rdm_path, save_format='npz')
    # Perform RSA
    return RSA_helper(save_path, brain_path, model_name, roi)


def brain_similarity_rsa_custom(model, model_name, my_preprocessor, my_cleaner, my_extractor, brain_path, feat_path, rdm_path, roi, device):
    # Set paths
    current_dir = os.getcwd()
    save_path = os.path.join(current_dir, f"{model_name}_RDM")
    # Load dataset
    dataset_path = DatasetNSD_872.load_dataset()
    stimuli_path = dataset_path["NSD_872_images"]
    if not os.path.isdir(save_path):
        # Extract features
        fx = FeatureExtractor(model=model, device=device, preprocessor=my_preprocessor, feature_cleaner=my_cleaner,
                              extraction_function=my_extractor)
        layers_to_extract = fx.get_all_layers()
        fx.extract(data_path=stimuli_path, save_path=feat_path, consolidate_per_layer=False,  layers_to_extract=layers_to_extract)
        # Create RDM of model
        creator = RDMCreator(verbose=True, device=device)
        save_path = creator.create_rdms(feature_path=feat_path, save_path=rdm_path, save_format='npz')
    # Perform RSA
    return RSA_helper(save_path, brain_path, model_name, roi)


def RSA_helper(save_path, brain_path, model_name, roi):
    # picks row with highest %R2 of specified ROI
    evaluation = RSA(save_path, brain_path, model_name=model_name)
    df = evaluation.evaluate()
    print(df["ROI"].unique())
    df_roi = df[df["ROI"] == roi]
    max_r2_value = df_roi['%R2'].max()
    df_max_r2 = df_roi[df_roi['%R2'] == max_r2_value]
    df_filtered = df_max_r2[['ROI', 'Layer', 'Model', '%R2']]
    return df_filtered.iloc[:1]
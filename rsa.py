import os
import torch
import shutil
import sys
from net2brain.evaluations.rsa import RSA
from net2brain.feature_extraction import FeatureExtractor
from net2brain.rdm_creation import RDMCreator
from utils.rsa_new import RSA
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm import create_model
import torchextractor as tx


def main(model_name, netset, roi_name, stimuli_path, dataset, rdm_save_path, num_subj, device="cuda" if torch.cuda.is_available() else "cpu"):
    current_dir = os.getcwd()
    rdm_path = f"{model_name}_RDM_{dataset}"
    save_path = os.path.join(current_dir, rdm_path)
    feat_path = f"{model_name}_Feat_{dataset}"
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
    RSA(save_path, brain_path,model_name,roi_name, dataset, num_subj)


def main_custom(model_name, roi_name, stimuli_path, dataset, rdm_save_path, num_subj, device="cuda" if torch.cuda.is_available() else "cpu"):
    current_dir = os.getcwd()
    rdm_path = f"{model_name}_RDM_{dataset}"
    save_path = os.path.join(current_dir, rdm_path)
    feat_path = f"{model_name}_Feat_{dataset}"
    if not os.path.isdir(save_path):
        # Extract features
        model = create_model(model_name, pretrained=True)
        fx = FeatureExtractor(model=model, device=device, preprocessor=my_preprocessor, feature_cleaner=my_cleaner,
                              extraction_function=my_extractor)
        layers_to_extract = fx.get_all_layers()
        fx.extract(data_path=stimuli_path, save_path=feat_path, consolidate_per_layer=False,  layers_to_extract=layers_to_extract)
        # Create RDM of model
        creator = RDMCreator(verbose=True, device=device)
        save_path = creator.create_rdms(feature_path=feat_path, save_path=rdm_path, save_format='npz')

    feat_path_complete = os.path.join(current_dir, feat_path)
    if os.path.exists(feat_path_complete):
        shutil.rmtree(feat_path_complete)

    brain_path = os.path.join(current_dir, rdm_save_path, roi_name)
    RSA(save_path, brain_path,model_name,roi_name, dataset, num_subj)


def my_preprocessor(image, model, device):
    """
    Args:
        image (Union[Image.Image, List[Image.Image]]): A PIL Image or a list of PIL Images.
        model (str): The name of the model, used to determine specific preprocessing if necessary.
        device (str): The device to which the tensor should be transferred ('cuda' for GPU, 'cpu' for CPU).

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: The preprocessed image(s) as PyTorch tensor(s).
    """

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
    models_list = ['vit_base_patch16_clip_224.openai', 'efficientnet_b3.ra2_in1k', 'vit_base_patch16_224.dino', 'beit_base_patch16_224.in22k_ft_in22k_in1k',
                   'gmlp_s16_224.ra3_in1k', 'vit_base_patch16_224.mae', 'convnext_base.fb_in22k_ft_in1k']
    model_name = models_list[num]

    stimuli_path = os.path.join(os.getcwd(),"NSD Dataset", f"NSD_872_images")
    rdm_save_path = "rdm"
    dataset = "rsa"
    main_custom(model_name, "V1", stimuli_path, dataset, rdm_save_path, 8)
    main_custom(model_name, "V2", stimuli_path, dataset, rdm_save_path, 8)
    main_custom(model_name, "V4", stimuli_path, dataset, rdm_save_path, 8)
    main_custom(model_name, "IT", stimuli_path, dataset, rdm_save_path, 8)

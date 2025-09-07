import os
import time

import torch
import shutil
import sys
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
        fx.extract(data_path=stimuli_path, save_path=feat_path, consolidate_per_layer=False,  layers_to_extract=layers_to_extract)
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
        model (str): The model
        device (str): The device to which the tensor should be transferred ('cuda' for GPU, 'cpu' for CPU).

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: The preprocessed image(s) as PyTorch tensor(s).
    """
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    img_tensor = transform(image)
    return img_tensor.unsqueeze(0)


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
                   'convnext_base.fb_in22k_ft_in1k',"tf_efficientnetv2_s.in21k_ft_in1k", 'resnetv2_50x1_bit.goog_in21k_ft_in1k',
                   'mixer_b16_224.goog_in21k_ft_in1k','mobilenetv3_large_100.miil_in21k_ft_in1k',
                   'fastvit_t8.apple_dist_in1k','mobilevit_s.cvnets_in1k','maxvit_nano_rw_256.sw_in1k']
    cornet = ["cornet_s", "cornet_z", "cornet_rt"]

    models_list = ["tf_efficientnetv2_s.in21k_ft_in1k", 'resnetv2_50x1_bit.goog_in21k_ft_in1k',
                   'mixer_b16_224.goog_in21k_ft_in1k','mobilenetv3_large_100.miil_in21k_ft_in1k',
                   'fastvit_t8.apple_dist_in1k','mobilevit_s.cvnets_in1k','maxvit_nano_rw_256.sw_in1k']

    model_name = models_list[num]

    stimuli_path = os.path.join(os.getcwd(),"NSD Dataset", f"NSD_872_images")
    rdm_save_path = os.path.join("rdms", "rdm_natural")
    dataset = "rsa_natural"

    if model_name in standard:
        main(model_name, "Standard", "V1", stimuli_path, dataset, rdm_save_path, 8)
        main(model_name, "Standard", "V2", stimuli_path, dataset, rdm_save_path, 8)
        main(model_name, "Standard", "V4", stimuli_path, dataset, rdm_save_path, 8)
        main(model_name, "Standard", "IT", stimuli_path, dataset, rdm_save_path, 8)
    elif model_name in timm:
        main(model_name, "Timm", "V1", stimuli_path, dataset, rdm_save_path, 8)
        main(model_name, "Timm", "V2", stimuli_path, dataset, rdm_save_path, 8)
        main(model_name, "Timm", "V4", stimuli_path, dataset, rdm_save_path, 8)
        main(model_name, "Timm", "IT", stimuli_path, dataset, rdm_save_path, 8)
    elif model_name in cornet:
        main(model_name, "Cornet", "V1", stimuli_path, dataset, rdm_save_path, 8)
        main(model_name, "Cornet", "V2", stimuli_path, dataset, rdm_save_path, 8)
        main(model_name, "Cornet", "V4", stimuli_path, dataset, rdm_save_path, 8)
        main(model_name, "Cornet", "IT", stimuli_path, dataset, rdm_save_path, 8)
    else:
        main_custom(model_name, "V1", stimuli_path, dataset, rdm_save_path, 8)
        main_custom(model_name, "V2", stimuli_path, dataset, rdm_save_path, 8)
        main_custom(model_name, "V4", stimuli_path, dataset, rdm_save_path, 8)
        main_custom(model_name, "IT", stimuli_path, dataset, rdm_save_path, 8)

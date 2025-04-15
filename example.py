import os
import torch
import shutil
import sys

from brain_similarity import brain_similarity_rsa
from net2brain.architectures.pytorch_models import Standard
from measure_accuracy import measure_accuracy
from effective_robustness import effective_robustness
from torchvision.transforms import transforms as trn


def main(model_name, netset):
    # set paths
    current_dir = os.getcwd()
    brain_path = os.path.join(current_dir, "NSD Dataset", "NSD_872_RDMs", "prf-visualrois", "combined")
    feat_path = f"{model_name}_Feat"
    rdm_path = f"{model_name}_RDM"
    # RSA
    df = brain_similarity_rsa(model_name, netset, brain_path, feat_path, rdm_path, "(1) V4_both_fmri")
    print(df.to_string())
    feat_path_complete = os.path.join(current_dir, f"{model_name}_Feat")
    if os.path.exists(feat_path_complete):
        shutil.rmtree(feat_path_complete)
    # Save to CSV
    csv_filename = 'rsa.csv'
    file_exists = os.path.isfile(csv_filename)
    df.to_csv(csv_filename, mode='a', index=False, header=not file_exists)


if __name__ == '__main__':
    num = int(sys.argv[1])
    models_list = ['ResNet50','AlexNet','Densenet121', 'Densenet161', 'Densenet169', 'Densenet201',
               'GoogleNet', 'ResNet101', 'ResNet152', 'ResNet18', 'ResNet34',
               'ShuffleNetV2x05', 'ShuffleNetV2x10', 'Squeezenet1_0', 'Squeezenet1_1',
               'VGG11', 'VGG11_bn', 'VGG13', 'VGG13_bn', 'VGG16',
               'VGG16_bn', 'VGG19', 'VGG19_bn', 'efficientnet_b0', 'efficientnet_b1',
               'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
               'mnasnet05', 'mnasnet10', 'mobilenet_v2',
               'mobilenet_v3_large', 'mobilenet_v3_small']
    model_name = models_list[num]
    main(model_name, "Standard")

import os
import shutil
import sys

from brain_similarity import brain_similarity_rsa


def main(model_name, netset, roi):
    # set paths
    current_dir = os.getcwd()
    brain_path = os.path.join(current_dir, "RDM", roi)
    feat_path = f"{model_name}_Feat"
    rdm_path = f"{model_name}_RDM"
    # RSA
    brain_similarity_rsa(model_name, netset, brain_path, feat_path, rdm_path, roi)
    feat_path_complete = os.path.join(current_dir, f"{model_name}_Feat")
    if os.path.exists(feat_path_complete):
        shutil.rmtree(feat_path_complete)


if __name__ == '__main__':
    num = int(sys.argv[1])

    """ models_list = ['ResNet50','AlexNet','Densenet121', 'Densenet161', 'Densenet169', 'Densenet201',
               'GoogleNet', 'ResNet101', 'ResNet152', 'ResNet18', 'ResNet34',
               'ShuffleNetV2x05', 'ShuffleNetV2x10', 'Squeezenet1_0', 'Squeezenet1_1',
               'VGG11', 'VGG11_bn', 'VGG13', 'VGG13_bn', 'VGG16',
               'VGG16_bn', 'VGG19', 'VGG19_bn', 'efficientnet_b0', 'efficientnet_b1',
               'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
               'mnasnet05', 'mnasnet10', 'mobilenet_v2',
               'mobilenet_v3_large', 'mobilenet_v3_small']"""
    models_list = ['inception_v3', 'inception_resnet_v2', 'xception',
                  'nasnetalarge', 'pnasnet5large', 'tf_efficientnet_b2_ns','tf_efficientnet_b4_ns'
                  'resnext50_32x4d', 'resnext101_32x8d',
                  'vit_base_patch16_224', 'vit_large_patch16_224', 'vit_huge_patch14_224_in21k',
                  'deit_base_patch16_224', 'swin_base_patch4_window7_224',
                  'mixer_b16_224', 'convnext_base', 'convnext_large',
                  'nfnet_l0', 'dm_nfnet_f0', 'regnety_032', 'regnety_080',
                  'cait_m36_384', 'coat_lite_mini','hrnet_w48', 'seresnet50',
                  'gluon_resnet50_v1c', 'gluon_resnext101_64x4d',
                  'wide_resnet50_2', 'wide_resnet101_2']
    model_name = models_list[num]
    main(model_name, "Timm", "V4")
    main(model_name, "Timm", "IT")


import sys
import os
from rsa import main
from rsa import main_custom

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
            'resnext50_32x4d', 'resnext101_32x8d', 'vit_base_patch16_224', 'vit_large_patch16_224', 'deit_base_patch16_224',
            'swin_base_patch4_window7_224', 'mixer_b16_224', 'nfnet_l0', 'dm_nfnet_f0', 'regnety_032',
            'regnety_080', 'coat_lite_mini','seresnet50', 'gluon_resnet50_v1c', 'gluon_resnext101_64x4d',
            'wide_resnet50_2', 'convit_small']
    timm_custom = ['efficientnet_b3.ra2_in1k', 'beit_base_patch16_224.in22k_ft_in22k_in1k', 'gmlp_s16_224.ra3_in1k', 'convnext_base.fb_in22k_ft_in1k']
    models_list = ['resnext101_32x8d', 'vit_base_patch16_224', 'vit_large_patch16_224',
                   'deit_base_patch16_224',
                   'swin_base_patch4_window7_224', 'mixer_b16_224', 'nfnet_l0', 'dm_nfnet_f0', 'regnety_032',
                   'regnety_080', 'coat_lite_mini', 'seresnet50', 'gluon_resnet50_v1c', 'gluon_resnext101_64x4d',
                   'wide_resnet50_2', 'convit_small',
                   'inception_v3', 'inception_resnet_v2', 'xception', 'tf_efficientnet_b2_ns', 'tf_efficientnet_b4_ns',
                   'resnext50_32x4d', 'resnext101_32x8d', 'vit_base_patch16_224', 'vit_large_patch16_224',
                   'deit_base_patch16_224',
                   'swin_base_patch4_window7_224', 'mixer_b16_224', 'nfnet_l0', 'dm_nfnet_f0', 'regnety_032',
                   'regnety_080', 'coat_lite_mini', 'seresnet50', 'gluon_resnet50_v1c', 'gluon_resnext101_64x4d',
                   'wide_resnet50_2', 'convit_small',
                   'efficientnet_b3.ra2_in1k', 'beit_base_patch16_224.in22k_ft_in22k_in1k', 'gmlp_s16_224.ra3_in1k', 'convnext_base.fb_in22k_ft_in1k']

    model_name = models_list[num]

    stimuli_path = os.path.join(os.getcwd(), "Illusion_Images")
    rdm_save_path = "rdm_illusion"
    dataset = "rsa_illusion"

    if model_name in standard:
        main(model_name, "Standard", "V1", stimuli_path, dataset, rdm_save_path, 6)
        main(model_name, "Standard", "V2", stimuli_path, dataset, rdm_save_path, 6)
        main(model_name, "Standard", "V4", stimuli_path, dataset, rdm_save_path, 6)
        main(model_name, "Standard", "IT", stimuli_path, dataset, rdm_save_path, 6)
    elif model_name in timm:
        main(model_name, "Timm", "V1", stimuli_path, dataset, rdm_save_path, 6)
        main(model_name, "Timm", "V2", stimuli_path, dataset, rdm_save_path, 6)
        main(model_name, "Timm", "V4", stimuli_path, dataset, rdm_save_path, 6)
        main(model_name, "Timm", "IT", stimuli_path, dataset, rdm_save_path, 6)
    else:
        main_custom(model_name, "V1", stimuli_path, dataset, rdm_save_path, 6)
        main_custom(model_name, "V2", stimuli_path, dataset, rdm_save_path, 6)
        main_custom(model_name, "V4", stimuli_path, dataset, rdm_save_path, 6)
        main_custom(model_name, "IT", stimuli_path, dataset, rdm_save_path, 6)
import os
import sys
from encoding import encoding_custom
from encoding import encoding


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
                   'convnext_base.fb_in22k_ft_in1k']
    cornet = ["cornet_s", "cornet_z"]

    models_list = ['ResNet50', 'AlexNet', 'Densenet121', 'Densenet161', 'Densenet169',
                    'Densenet201', 'GoogleNet', 'ResNet101', 'ResNet152', 'ResNet18',
                    'ResNet34', 'ShuffleNetV2x05', 'ShuffleNetV2x10', 'Squeezenet1_0', 'Squeezenet1_1',
                    'VGG11', 'VGG11_bn', 'VGG13', 'VGG13_bn', 'VGG16',
                    'VGG16_bn', 'VGG19', 'VGG19_bn', 'efficientnet_b0', 'efficientnet_b1',
                    'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'mnasnet05',
                    'mnasnet10', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
                    'inception_v3', 'inception_resnet_v2', 'xception', 'tf_efficientnet_b2_ns', 'tf_efficientnet_b4_ns',
                    'resnext50_32x4d', 'resnext101_32x8d', 'vit_base_patch16_224', 'vit_large_patch16_224',
                    'deit_base_patch16_224',
                    'swin_base_patch4_window7_224', 'mixer_b16_224', 'nfnet_l0', 'dm_nfnet_f0', 'regnety_032',
                    'regnety_080', 'coat_lite_mini', 'seresnet50', 'gluon_resnet50_v1c', 'gluon_resnext101_64x4d',
                    'wide_resnet50_2', 'convit_small',
                    'efficientnet_b3.ra2_in1k', 'beit_base_patch16_224.in22k_ft_in22k_in1k', 'gmlp_s16_224.ra3_in1k',
                    'convnext_base.fb_in22k_ft_in1k',
                    "cornet_s", "cornet_z"]

    model_name = models_list[num]

    stimuli_path = os.path.join(os.getcwd(), "Things_Images")
    fmri_dataset = "fmri_things"
    save_folder = "encoding_things"
    if model_name in standard:
        encoding(model_name, "Standard", "V1", stimuli_path, fmri_dataset, save_folder, 2)
        encoding(model_name, "Standard", "V2", stimuli_path, fmri_dataset, save_folder, 2)
        encoding(model_name, "Standard", "V4", stimuli_path, fmri_dataset, save_folder, 2)
        encoding(model_name, "Standard", "IT", stimuli_path, fmri_dataset, save_folder, 2)
    elif model_name in timm:
        encoding(model_name, "Timm", "V1", stimuli_path, fmri_dataset, save_folder, 2)
        encoding(model_name, "Timm", "V2", stimuli_path, fmri_dataset, save_folder, 2)
        encoding(model_name, "Timm", "V4", stimuli_path, fmri_dataset, save_folder, 2)
        encoding(model_name, "Timm", "IT", stimuli_path, fmri_dataset, save_folder, 2)
    elif model_name in cornet:
        encoding(model_name, "Cornet", "V1", stimuli_path, fmri_dataset, save_folder, 2)
        encoding(model_name, "Cornet", "V2", stimuli_path, fmri_dataset, save_folder, 2)
        encoding(model_name, "Cornet", "V4", stimuli_path, fmri_dataset, save_folder, 2)
        encoding(model_name, "Cornet", "IT", stimuli_path, fmri_dataset, save_folder, 2)
    else:
        features = encoding_custom(model_name, "V1", stimuli_path, fmri_dataset, save_folder, 2)
        encoding_custom(model_name, "V2", stimuli_path, fmri_dataset, save_folder, 2, features)
        encoding_custom(model_name, "V4", stimuli_path, fmri_dataset, save_folder, 2, features)
        encoding_custom(model_name, "IT", stimuli_path, fmri_dataset, save_folder, 2, features)
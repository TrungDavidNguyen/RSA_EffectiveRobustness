import os
import torch
import shutil
import sys

from brain_similarity import brain_similarity_rsa
from net2brain.architectures.timm_models import Timm
from effective_robustness import measure_accuracy
from effective_robustness import effective_robustness


def main(model_name):
    # set paths
    current_dir = os.getcwd()
    brain_path = os.path.join(current_dir, "NSD Dataset", "NSD_872_RDMs", "prf-visualrois", "combined")
    feat_path = f"{model_name}_Feat"
    rdm_path = f"{model_name}_RDM"
    # RSA
    df = brain_similarity_rsa(model_name, "Timm", brain_path, feat_path,rdm_path, "(1) V4_both_fmri")
    print(df.to_string())
    feat_path_complete = os.path.join(current_dir, f"{model_name}_Feat")
    if os.path.exists(feat_path_complete):
        shutil.rmtree(feat_path_complete)

    # measure accuracy
    # get model and transform
    models = Timm(model_name, "cuda" if torch.cuda.is_available() else "cpu")
    model = models.get_model(pretrained=True)
    transform = model.create_transform()
    """
        transform = trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    """
    imagenet_path = os.path.join(os.getcwd(), "imagenet")
    id_accuracy = measure_accuracy(model, imagenet_path, transform)
    print(model_name," id accuracy", id_accuracy)
    sketch_path = os.path.join(os.getcwd(), "imagenet_sketch")
    ood_accuracy = measure_accuracy(model, sketch_path, transform)
    print(model_name," ood accuracy", ood_accuracy)
    # values for imagenet sketch
    intercept = -2.370072912552283
    slope = 1.0709154135668684
    # values for imagenet r
    # intercept = -1.5999151525728197
    # slope = 0.9115905266235703
    eff_robustness = effective_robustness(id_accuracy, ood_accuracy, intercept, slope)
    print(model_name, "eff robust", eff_robustness)
    df.loc[:, "eff.Robustness"] = eff_robustness
    df.loc[:, "imagenet1k"] = id_accuracy
    df.loc[:, "imagenet-sketch"] = ood_accuracy

    # Save to CSV
    csv_filename = 'results-imagenet-sketch.csv'
    file_exists = os.path.isfile(csv_filename)
    df.to_csv(csv_filename, mode='a', index=False, header=not file_exists)


if __name__ == '__main__':
    num = int(sys.argv[1])
    """
        models_list = ['ResNet50','AlexNet','Densenet121', 'Densenet161', 'Densenet169', 'Densenet201',
                   'GoogleNet', 'ResNet101', 'ResNet152', 'ResNet18', 'ResNet34',
                   'ShuffleNetV2x05', 'ShuffleNetV2x10', 'Squeezenet1_0', 'Squeezenet1_1',
                   'VGG11', 'VGG11_bn', 'VGG13', 'VGG13_bn', 'VGG16',
                   'VGG16_bn', 'VGG19', 'VGG19_bn', 'efficientnet_b0', 'efficientnet_b1',
                   'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                   'efficientnet_b6', 'efficientnet_b7', 'mnasnet05', 'mnasnet10', 'mobilenet_v2',
                   'mobilenet_v3_large', 'mobilenet_v3_small', 'mobilenet_v2', 'mobilenet_v3_large',
                   'mobilenet_v3_small', 'resnext101_32x8d', 'resnext50_32x4d', 'wide_resnet101_2', 'wide_resnet50_2']
    """
    models_list = ['tf_efficientnet_l2_ns','xception','vit_base_patch16_224', 'vit_base_patch16_224_in21k',
                   'vit_base_r50_s16_224_in21k', 'vit_huge_patch14_224_in21k', 'vit_large_patch16_224',
                   'vit_large_patch16_224_in21k', 'vit_small_patch16_224','vit_tiny_patch16_224']
    model_name = models_list[num]
    main(model_name)

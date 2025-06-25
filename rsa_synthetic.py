import sys
import os
from rsa import main
from rsa import main_custom

if __name__ == '__main__':
    num = int(sys.argv[1])
    models_list = ['vit_base_patch16_clip_224.openai', 'efficientnet_b3.ra2_in1k', 'vit_base_patch16_224.dino',
                   'beit_base_patch16_224.in22k_ft_in22k_in1k',
                   'gmlp_s16_224.ra3_in1k', 'vit_base_patch16_224.mae', 'convnext_base.fb_in22k_ft_in1k']
    model_name = models_list[num]

    stimuli_path = os.path.join(os.getcwd(), "NSD Synthetic", f"NSD_284_images")
    rdm_save_path = os.path.join("rdms", "rdm_synthetic")
    dataset = "rsa_synthetic"
    main_custom(model_name, "V1", stimuli_path, dataset, rdm_save_path, 8)
    main_custom(model_name, "V2", stimuli_path, dataset, rdm_save_path, 8)
    main_custom(model_name, "V4", stimuli_path, dataset, rdm_save_path, 8)
    main_custom(model_name, "IT", stimuli_path, dataset, rdm_save_path, 8)
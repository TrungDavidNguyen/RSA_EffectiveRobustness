import os
import sys
from encoding import encoding_custom
from encoding import encoding


if __name__ == '__main__':
    num = int(sys.argv[1])
    models_list = ['efficientnet_b3.ra2_in1k', 'beit_base_patch16_224.in22k_ft_in22k_in1k',
                   'gmlp_s16_224.ra3_in1k', 'convnext_base.fb_in22k_ft_in1k']
    model_name = models_list[num]

    stimuli_path = os.path.join(os.getcwd(), "NSD Synthetic", "NSD_284_images")
    fmri_dataset = "fmri_synthetic"
    save_folder = "encoding_synthetic"

    features = encoding_custom(model_name,  "V1", stimuli_path, fmri_dataset, save_folder, 8)
    encoding_custom(model_name, "V2", stimuli_path, fmri_dataset, save_folder, 8, features)
    encoding_custom(model_name, "V4", stimuli_path, fmri_dataset, save_folder, 8, features)
    encoding_custom(model_name, "IT", stimuli_path, fmri_dataset, save_folder, 8, features)
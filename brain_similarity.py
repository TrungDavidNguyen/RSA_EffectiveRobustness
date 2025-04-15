import os
import torch
import pandas as pd
from net2brain.evaluations.rsa import RSA
from net2brain.utils.download_datasets import DatasetNSD_872
from net2brain.feature_extraction import FeatureExtractor
from net2brain.rdm_creation import RDMCreator


def brain_similarity_rsa(model_name, netset, brain_path, feat_path, rdm_path, roi, device="cuda" if torch.cuda.is_available() else "cpu"):
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
    evaluation = RSA(save_path, brain_path, model_name=model_name)
    df = evaluation.evaluate()

    os.makedirs(f"rsa/{roi}", exist_ok=True)
    csv_filename = f"rsa/{roi}/{model_name}_RSA.csv"
    df.to_csv(csv_filename, index=False)
    # get highest %R2 value
    max_r2_value = df['%R2'].max()

    csv_filename = 'results/rsa.csv'

    if os.path.exists(csv_filename):
        df_new = pd.read_csv(csv_filename)
        if model_name in df_new['Model'].values:
            if f"%R2_{roi}" not in df_new.columns:
                df_new[f"%R2_{roi}"] = None
            df_new.loc[df_new['Model'] == model_name, f"%R2_{roi}"] = max_r2_value
        else:
            new_row = pd.Series({col: None for col in df_new.columns})
            new_row['Model'] = model_name
            new_row[f"%R2_{roi}"] = max_r2_value
            df_new = pd.concat([df_new, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df_new = pd.DataFrame({"Model": [model_name], f"%R2_{roi}": [max_r2_value]})
    df_new.to_csv(csv_filename, index=False)

    # get rows with highest %R2 value
    df_max_r2 = df[df['%R2'] == max_r2_value]
    # take one
    df_max_r2 = df_max_r2.iloc[:1]
    csv_filename = f"rsa/{roi}/results-RSA-{roi}.csv"
    file_exists = os.path.isfile(csv_filename)
    df_max_r2.to_csv(csv_filename, mode='a', index=False, header=not file_exists)

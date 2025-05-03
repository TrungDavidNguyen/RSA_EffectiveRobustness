import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy import stats
from net2brain.evaluations.noiseceiling import NoiseCeiling
from net2brain.evaluations.eval_helper import natural_keys

def RSA(model_rdms_path, brain_rdms_path, model_name, roi_name):
    """
    :param model_rdms_path: path to folder with rdms of all layers
    :param brain_rdms_path: path to folder with rdm npz file
    :param model_name: name of model
    :return:
    """
    name = "rsa_synthetic"
    # list of rdm file names of each layer
    model_rdms = folderlookup(model_rdms_path)
    model_rdms.sort(key=natural_keys)
    # list of rdm file names of each subject
    brain_rdms = folderlookup(brain_rdms_path)
    brain_rdms_path_full = os.path.join(brain_rdms_path, brain_rdms[0])

    noise_ceiling_calc = NoiseCeiling(brain_rdms[0], brain_rdms_path_full, "spearman", True)
    this_nc = noise_ceiling_calc.noise_ceiling()
    lnc = this_nc["lnc"]
    unc = this_nc["unc"]

    num_subj = 8
    all_dicts = [[] for _ in range(num_subj + 1)]

    for counter, layer in enumerate(model_rdms):
        model_rdm_npz = np.load(os.path.join(model_rdms_path, layer))
        model_rdm_npz = check_squareform(model_rdm_npz["rdm"])
        brain_rdms_npz = np.load(brain_rdms_path_full)["rdm"]
        corr = model_spearman(model_rdm_npz, brain_rdms_npz)
        corr_list = np.square(corr)
        # Take mean
        r = np.mean(corr_list)

        # ttest: Ttest_1sampResult(statistic=3.921946, pvalue=0.001534)
        significance = stats.ttest_1samp(corr_list, 0)[1]

        # standard error of mean
        sem = stats.sem(corr_list)  # standard error of mean

        # Create dictionary to save data
        layer_key = "(" + str(counter) + ") " + layer
        for i in range(num_subj):
            output_dict = {
                "ROI": [roi_name],
                "Model": [model_name],
                "Layer": [layer_key],
                "R2": [corr_list[i]],
                "%R2": [(corr_list[i] / lnc) * 100.],
                "Significance": [significance],
                "SEM": [sem],
                "LNC": [lnc],
                "UNC": [unc]
            }
            all_dicts[i].append(output_dict)
        # Add mean as last element
        output_dict = {
                        "ROI": [roi_name],
                        "Model": [model_name],
                        "Layer": [layer_key],
                        "R2": [r],
                        "%R2": [(r / lnc) * 100.],
                        "Significance": [significance],
                        "SEM": [sem],
                        "LNC": [lnc],
                        "UNC": [unc]
                        }
        all_dicts[num_subj].append(output_dict)
    # all subjects
    for i in range(num_subj):
        all_rois_df = pd.DataFrame(
            columns=['ROI', 'Layer', "Model", "R2", "%R2", 'Significance', 'SEM', 'LNC', 'UNC'])
        for layer_dict in all_dicts[i]:
            layer_df = pd.DataFrame.from_dict(layer_dict)
            all_rois_df = pd.concat([all_rois_df, layer_df], ignore_index=True)
        os.makedirs(f"{name}/{roi_name}/rsa_{roi_name}_subj{i+1}", exist_ok=True)
        csv_filename = f"{name}/{roi_name}/rsa_{roi_name}_subj{i+1}/{model_name}_RSA.csv"
        all_rois_df.to_csv(csv_filename, index=False)

        all_rois_df = all_rois_df.loc[[all_rois_df['%R2'].idxmax()]]
        csv_filename = f'{name}/{roi_name}/rsa_{roi_name}_subj{i+1}/results-rsa-{roi_name}.csv'
        file_exists = os.path.isfile(csv_filename)
        all_rois_df.to_csv(csv_filename, mode='a', index=False, header=not file_exists)
    # mean of all subjects
    all_rois_df = pd.DataFrame(
        columns=['ROI', 'Layer', "Model", "R2", "%R2", 'Significance', 'SEM', 'LNC', 'UNC'])
    for layer_dict in all_dicts[num_subj]:
        layer_df = pd.DataFrame.from_dict(layer_dict)
        all_rois_df = pd.concat([all_rois_df, layer_df], ignore_index=True)
    os.makedirs(f"{name}/{roi_name}/rsa_{roi_name}_mean", exist_ok=True)
    csv_filename = f"{name}/{roi_name}/rsa_{roi_name}_mean/{model_name}_RSA.csv"
    all_rois_df.to_csv(csv_filename, index=False)

    max_r2_value = all_rois_df['%R2'].max()

    csv_filename = f'results/{name}.csv'

    if os.path.exists(csv_filename):
        df_new = pd.read_csv(csv_filename)
        if model_name in df_new['Model'].values:
            if f"%R2_{roi_name}" not in df_new.columns:
                df_new[f"%R2_{roi_name}"] = None
            df_new.loc[df_new['Model'] == model_name, f"%R2_{roi_name}"] = max_r2_value
        else:
            new_row = pd.Series({col: None for col in df_new.columns})
            new_row['Model'] = model_name
            new_row[f"%R2_{roi_name}"] = max_r2_value
            df_new = pd.concat([df_new, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df_new = pd.DataFrame({"Model": [model_name], f"%R2_{roi_name}": [max_r2_value]})
    df_new.to_csv(csv_filename, index=False)

    all_rois_df = all_rois_df.loc[[all_rois_df['%R2'].idxmax()]]
    csv_filename = f'{name}/{roi_name}/rsa_{roi_name}_mean/results-rsa-{roi_name}.csv'
    file_exists = os.path.isfile(csv_filename)
    all_rois_df.to_csv(csv_filename, mode='a', index=False, header=not file_exists)
def folderlookup(path):
    """Looks at the available files and returns the chosen one
    Args:
        path (str/path): path to folder
    Returns:
        list: list of files in dir
    """

    files = os.listdir(path)  # Which folders do we have?
    file_sets = []

    for f in files:
        if ".json" not in f and ".DS_Store" not in f:
            if f != ".ipynb_checkpoints":
                file_sets.append(f)

    return file_sets
def check_squareform(rdm):
        """Ensure that the RDM is in squareform.

        Args:
            rdm (numpy array): The RDM in either squareform or vector form.

        Returns:
            numpy array: The RDM in squareform.
        """
        # Check if the RDM is in squareform. If the array is 2D and square, it's already in squareform.
        if rdm.ndim == 2 and rdm.shape[0] == rdm.shape[1]:
            return rdm
        else:
            # Convert to squareform.
            return squareform(rdm)


def sq(x):
    """Converts a square-form distance matrix from a vector-form distance vector

    Args:
        x (numpy array): numpy array that should be vector

    Returns:
        numpy array: numpy array as vector
    """
    if x.ndim == 2:  # Only apply to 2D inputs
        return squareform(x, force='tovector', checks=False)
    return x  # If already 1D, return as-is


def model_spearman(model_rdm, rdms):
    """Calculate Spearman correlation."""
    model_rdm_sq = sq(model_rdm)
    return [stats.spearmanr(sq(rdm), model_rdm_sq)[0] for rdm in rdms]
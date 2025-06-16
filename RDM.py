import numpy as np
import os
import torch
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


def generate_RDMs(rois, roi_name, dataset):
    images = 0
    if dataset == "NSD Dataset":
        images = 872
        folder = "rdm"
    elif dataset == "NSD Synthetic":
        images = 284
        folder = "rdm_synthetic"

    all_rdms = []
    for subjects in range(1, 9):
        fmri = None
        for roi in rois:
            for hemisphere in ["lh","rh"]:
                group = ""
                if roi in ["EBA", "FBA-1", "FBA-2"]:
                    group = "floc-bodies"
                elif roi in ["FFA-1", "FFA-2", "OFA"]:
                    group = "floc-faces"
                elif roi in ["PPA", "OPA", "RSC"]:
                    group = "floc-places"
                elif roi in ["hV4", "V1d", "V1v", "V2d", "V2v", "V3d", "V3v",]:
                    group = "prf-visualrois"
                current_dir = os.getcwd()
                roi_path = os.path.join(current_dir,dataset,f"NSD_{images}_fmri", group, roi, f"subj0{subjects}_roi-{roi}_{hemisphere}.npy")
                if fmri is None:
                    try:
                        fmri = np.load(roi_path).astype(np.float64)
                    except FileNotFoundError:
                        pass
                else:
                    try:
                        fmri = np.concatenate((fmri, np.load(roi_path).astype(np.float64)), axis=1)
                    except FileNotFoundError:
                        pass
        all_rdms.append(squareform(pdist(torch.from_numpy(fmri), metric='correlation')))
    os.makedirs(f"{folder}/{roi_name}", exist_ok=True)
    np.savez(f"{folder}/{roi_name}/{roi_name}_both_fmri.npz", rdm=np.stack(all_rdms))


def generate_RDMs_illusion(rois, roi_name):
    all_rdms = []
    for subjects in [1,2,3,5,6,7]:
        fmri = None
        for roi in rois:
            current_dir = os.getcwd()
            roi_path = os.path.join(current_dir, "fmri_illusion", roi, f"{roi}_fmri_subj{subjects}",f'{roi}_both_subj{subjects}.npy')
            if fmri is None:
                try:
                    fmri = np.load(roi_path).astype(np.float64)
                except FileNotFoundError:
                    pass
            else:
                try:
                    fmri = np.concatenate((fmri, np.load(roi_path).astype(np.float64)), axis=1)
                except FileNotFoundError:
                    pass
        all_rdms.append(squareform(pdist(torch.from_numpy(fmri), metric='correlation')))
    os.makedirs(f"rdm_illusion/{roi_name}", exist_ok=True)
    np.savez(f"rdm_illusion/{roi_name}/{roi_name}_both_fmri.npz", rdm=np.stack(all_rdms))


def generate_RDMs_things(rois, roi_name):
    all_rdms = []
    for subjects in range(1, 3):
        fmri = None
        for roi in rois:
            current_dir = os.getcwd()
            roi_path = os.path.join(current_dir, "fmri_things", roi, f"{roi}_fmri_subj{subjects}",
                                    f'{roi}_both_subj{subjects}.npy')
            if fmri is None:
                try:
                    fmri = np.load(roi_path).astype(np.float64)
                except FileNotFoundError:
                    pass
            else:
                try:
                    fmri = np.concatenate((fmri, np.load(roi_path).astype(np.float64)), axis=1)
                except FileNotFoundError:
                    pass
        all_rdms.append(squareform(pdist(torch.from_numpy(fmri), metric='correlation')))
    os.makedirs(f"rdm_things/{roi_name}", exist_ok=True)
    np.savez(f"rdm_things/{roi_name}/{roi_name}_both_fmri.npz", rdm=np.stack(all_rdms))


def copy_RDM(roi):
    current_dir = os.getcwd()
    group = ""
    if roi in ["EBA", "FBA-1", "FBA-2"]:
        group = "floc-bodies"
    elif roi in ["FFA-1", "FFA-2", "OFA"]:
        group = "floc-faces"
    elif roi in ["PPA", "OPA", "RSC"]:
        group = "floc-places"
    elif roi in ["V4", "V1", "V2", "V3"]:
        group = "prf-visualrois"
    all_rdms = []
    for subj in range(1, 9):
        src = os.path.join(current_dir, "NSD Dataset", "NSD_872_RDMs", group,
                           f"subject_{subj}_merged_{roi}_both.npz")
        all_rdms.append(np.load(src)["rdm"])
    os.makedirs(f"rdm/{roi}", exist_ok=True)
    np.savez(f"rdm/{roi}/{roi}_both_fmri.npz", rdm=np.stack(all_rdms))


if __name__ == '__main__':
    #copy_RDM("V1")
    #copy_RDM("V2")
    #copy_RDM("V4")
    #generate_RDMs(["EBA", "FFA-1", "FFA-2", "FBA-1", "FBA-2", "PPA"], "IT", "NSD Dataset")
    generate_RDMs_things(["V1"],"V1")
    generate_RDMs_things(["V2"],"V2")
    generate_RDMs_things(["V4"],"V4")
    generate_RDMs_things(["IT"],"IT")


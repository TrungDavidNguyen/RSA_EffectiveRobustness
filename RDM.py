import numpy as np
import os
import torch
from scipy.spatial.distance import pdist
import shutil
from scipy.spatial.distance import squareform


def generate_IT_RDM():
    rois = {"EBA": "floc-bodies", "FFA-1": "floc-faces", "FFA-2": "floc-faces", "FBA-1": "floc-bodies", "FBA-2": "floc-bodies", "PPA": "floc-places"}
    all_rdms = []
    for subjects in range(1, 9):
        IT_fmri = None
        for roi in rois.keys():
            for hemisphere in ["lh","rh"]:
                group = rois[roi]
                current_dir = os.getcwd()
                roi_path = os.path.join(current_dir,"NSD Dataset","NSD_872_fmri",group,roi,f"subj0{subjects}_roi-{roi}_{hemisphere}.npy")
                if IT_fmri is None:
                    try:
                        IT_fmri = np.load(roi_path).astype(np.float64)
                    except FileNotFoundError:
                        pass
                else:
                    try:
                        IT_fmri = np.concatenate((IT_fmri, np.load(roi_path).astype(np.float64)), axis=1)
                    except FileNotFoundError:
                        pass
        all_rdms.append(squareform(pdist(torch.from_numpy(IT_fmri), metric='correlation')))
    os.makedirs(f"rdm/IT", exist_ok=True)
    np.savez(f"rdm/IT/IT_both_fmri.npz", rdm=np.stack(all_rdms))


def generate_RDM(roi):
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
    generate_IT_RDM()
    generate_RDM("V4")
    print(np.load(r"C:\Users\david\Desktop\RSA_EffectiveRobustness\rdm\V4\V4_both_fmri.npz")["rdm"].shape)
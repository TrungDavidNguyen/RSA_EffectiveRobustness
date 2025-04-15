import numpy as np
import os
import torch
from net2brain.rdm.dist import correlation
import shutil


def generate_IT_RDM():
    rois = ["EBA", "FFA-1", "FFA-2", "FBA-1", "FBA-2", "PPA"]
    rdms = []
    for subjects in range(1, 9):
        IT_fmri = None
        for roi in rois:
            for hemisphere in ["lh","rh"]:
                group = ""
                if roi in ["EBA", "FBA-1", "FBA-2"]:
                    group = "floc-bodies"
                if roi in ["FFA-1", "FFA-2"]:
                    group = "floc-faces"
                if roi == "PPA":
                    group = "floc-places"
                current_dir = os.getcwd()
                roi_path = os.path.join(current_dir,"NSD Dataset","NSD_872_fmri",group,roi,f"subj0{subjects}_roi-{roi}_{hemisphere}.npy")
                if IT_fmri is None:
                    try:
                        IT_fmri = np.load(roi_path)
                    except FileNotFoundError:
                        pass
                else:
                    try:
                        IT_fmri = np.concatenate((IT_fmri, np.load(roi_path)), axis=1)
                    except FileNotFoundError:
                        pass
        rdms.append(correlation(torch.from_numpy(IT_fmri)))
    rdms_combined = np.stack(rdms, axis=0)
    os.makedirs("RDM/IT", exist_ok=True)
    np.savez("RDM/IT/IT_both_fmri.npz", rdm=rdms_combined)


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

    src = os.path.join(current_dir, "NSD Dataset", "NSD_872_RDMs", group, "combined",
                       f"{roi}_both_fmri.npz")
    dst_dir = os.path.join("RDM", roi)
    dst = os.path.join(dst_dir, f"{roi}_both_fmri.npz")

    os.makedirs(dst_dir, exist_ok=True)
    shutil.copyfile(src, dst)


if __name__ == '__main__':
    copy_RDM("V4")
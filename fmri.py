import numpy as np
import os


def generate_fmri(rois, roi_name):
    for subjects in range(1, 9):
        fmri = None
        for roi in rois:
            for hemisphere in ["lh", "rh"]:
                group = ""
                if roi in ["EBA", "FBA-1", "FBA-2"]:
                    group = "floc-bodies"
                if roi in ["FFA-1", "FFA-2", "OFA"]:
                    group = "floc-faces"
                if roi in ["PPA", "OPA", "RSC"]:
                    group = "floc-places"
                if roi in ["hV4", "V1d", "V1v", "V2d", "V2v", "V3d", "V3v"]:
                    group = "prf-visualrois"
                current_dir = os.getcwd()
                roi_path = os.path.join(current_dir, "NSD Dataset", "NSD_872_fmri", group, roi, f"subj0{subjects}_roi-{roi}_{hemisphere}.npy")
                if fmri is None:
                    try:
                        fmri = np.load(roi_path)
                    except FileNotFoundError:
                        pass
                else:
                    try:
                        fmri = np.concatenate((fmri, np.load(roi_path)), axis=1)
                    except FileNotFoundError:
                        pass
        os.makedirs(f'fmri/{roi_name}/{roi_name}_fmri_subj{subjects}', exist_ok=True)
        np.save(f'fmri/{roi_name}/{roi_name}_fmri_subj{subjects}/{roi_name}_both_fmri_subj{subjects}.npy', fmri)

def generate_fmri_synthetic(rois, roi_name):
    for subjects in range(1, 9):
        fmri = None
        for roi in rois:
            for hemisphere in ["lh", "rh"]:
                group = ""
                if roi in ["EBA", "FBA-1", "FBA-2"]:
                    group = "floc-bodies"
                if roi in ["FFA-1", "FFA-2", "OFA"]:
                    group = "floc-faces"
                if roi in ["PPA", "OPA", "RSC"]:
                    group = "floc-places"
                if roi in ["hV4", "V1d", "V1v", "V2d", "V2v", "V3d", "V3v"]:
                    group = "prf-visualrois"
                current_dir = os.getcwd()
                roi_path = os.path.join(current_dir, "NSD Synthetic", "NSD_284_fmri", group, roi, f"subj0{subjects}_roi-{roi}_{hemisphere}.npy")
                if fmri is None:
                    try:
                        fmri = np.load(roi_path)
                    except FileNotFoundError:
                        pass
                else:
                    try:
                        fmri = np.concatenate((fmri, np.load(roi_path)), axis=1)
                    except FileNotFoundError:
                        pass
        os.makedirs(f'fmri_synthetic/{roi_name}/{roi_name}_fmri_subj{subjects}', exist_ok=True)
        np.save(f'fmri_synthetic/{roi_name}/{roi_name}_fmri_subj{subjects}/{roi_name}_both_fmri_subj{subjects}.npy', fmri)


def generate_fmri_illusion(rois, roi_name):
    for subjects in [1,2,3,5,6,7]:
        fmri = None
        for roi in rois:
            current_dir = os.getcwd()
            roi_path = os.path.join(current_dir, "fmri_illusion", roi, f"{roi}_fmri_subj{subjects}", f"{roi}_both_subj{subjects}.npy")
            if fmri is None:
                try:
                    fmri = np.load(roi_path)
                except FileNotFoundError:
                    pass
            else:
                try:
                    fmri = np.concatenate((fmri, np.load(roi_path)), axis=1)
                except FileNotFoundError:
                    pass
        os.makedirs(os.path.join("fmri_illusion", roi_name, f"{roi_name}_fmri_subj{subjects}"), exist_ok=True)
        np.save(os.path.join("fmri_illusion", roi_name, f"{roi_name}_fmri_subj{subjects}", f"{roi_name}_both_subj{subjects}.npy"), fmri)


def generate_fmri_things(rois, roi_name):
    for subjects in range(1, 3):
        fmri = None
        for roi in rois:
            current_dir = os.getcwd()
            roi_path = os.path.join(current_dir, "Things_test", "Things_fMRI_test",f"sub0{subjects}_test_roi-{roi}.npy")
            if fmri is None:
                try:
                    temp = np.load(roi_path)
                    # average each 10 cols and transpose
                    fmri = temp.reshape(temp.shape[0], -1, 10).mean(axis=2).T
                except FileNotFoundError:
                    pass
            else:
                try:
                    temp = np.load(roi_path)
                    fmri = np.concatenate((fmri, temp.reshape(temp.shape[0], -1, 10).mean(axis=2).T), axis=1)
                except FileNotFoundError:
                    pass
        os.makedirs(os.path.join("fmri_things", roi_name, f"{roi_name}_fmri_subj{subjects}"), exist_ok=True)
        np.save(os.path.join("fmri_things", roi_name, f"{roi_name}_fmri_subj{subjects}", f"{roi_name}_both_subj{subjects}.npy"), fmri)


if __name__ == '__main__':
    rois = ["rFFA","lFFA","rPPA","lPPA"]
    roi_name = "IT"
    generate_fmri_things(rois, roi_name)

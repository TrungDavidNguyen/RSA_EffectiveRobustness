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

if __name__ == '__main__':
    rois = ["hV4"]
    roi_name = "V4"
    generate_fmri(rois, roi_name)
    rois = ["EBA", "FBA-1", "FBA-2", "FFA-1", "FFA-2", "PPA"]
    roi_name = "IT"
    generate_fmri(rois, roi_name)
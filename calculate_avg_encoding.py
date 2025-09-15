import pandas as pd
import glob
import os
import plotting_thesis.utils as utils

evaluations = utils.PlottingConfig.EVALUATIONS_DICT["encoding"]
rois = utils.PlottingConfig.ROIS

for eval in evaluations:
    for roi in rois:
        base_path = os.path.join(eval, roi)
        subject_folders = sorted(glob.glob(os.path.join(base_path, f"encoding_{roi}_subj*")))
        if not subject_folders:
            continue

        csv_files = [os.path.basename(f) for f in glob.glob(os.path.join(subject_folders[0], "*.csv"))]

        output_dir = os.path.join(base_path, f"encoding_{roi}_mean")
        os.makedirs(output_dir, exist_ok=True)

        for csv_file in csv_files:
            dfs = []
            for folder in subject_folders:
                file_path = os.path.join(folder, csv_file)
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path)
                    except:
                        continue
                    df = df[["Layer", "Model", "R"]]
                    dfs.append(df)

            if dfs:
                all_df = pd.concat(dfs, axis=0)

                avg_df = (
                    all_df.groupby(["Layer", "Model"], as_index=False, sort=False)
                    .mean(numeric_only=True)
                )
                avg_df.to_csv(os.path.join(output_dir, csv_file), index=False)


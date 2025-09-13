import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# --- Configuration ---
# Use a raw string (r"...") or forward slashes for file paths to avoid issues.
BASE_PATH = r"C:\Users\david\Desktop\RSA_EffectiveRobustness\results"

EVALUATIONS = [
    "encoding_imagenet", "encoding_synthetic", "encoding_illusion", "encoding_natural",
    "rsa_synthetic", "rsa_illusion", "rsa_imagenet", "rsa_natural"
]

# Define a standard set of column names to unify the data
UNIFIED_SCORE_COLS = ['Score_V1', 'Score_V2', 'Score_V4', 'Score_IT']


def assign_group(eval_name: str) -> str:
    """Assigns a group name based on keywords in the evaluation name."""
    if "natural" in eval_name:
        return "natural"
    if "synthetic" in eval_name:
        return "synthetic"
    if "imagenet" in eval_name:
        return "imagenet"
    if "illusion" in eval_name:
        return "illusion"
    return "other"


if __name__ == '__main__':
    all_results = []
    scaler = MinMaxScaler()

    # --- Data Loading and Preprocessing Loop ---
    for eval_name in EVALUATIONS:
        print(f"Processing: {eval_name}...")

        # Determine the correct score columns based on the evaluation type
        if "rsa" in eval_name:
            original_score_cols = ['%R2_V1', '%R2_V2', '%R2_V4', '%R2_IT']
        elif "encoding" in eval_name:
            original_score_cols = ['R_V1', 'R_V2', 'R_V4', 'R_IT']
        else:
            print(f"Warning: Skipping '{eval_name}' - unknown type.")
            continue

        file_path = os.path.join(BASE_PATH, f"{eval_name}.csv")

        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Warning: File not found at {file_path}. Skipping.")
            continue

        # Create a mapping to rename original columns to unified names
        col_rename_map = dict(zip(original_score_cols, UNIFIED_SCORE_COLS))
        df.rename(columns=col_rename_map, inplace=True)

        # Ensure only existing columns are processed
        cols_to_process = [col for col in UNIFIED_SCORE_COLS if col in df.columns]
        if not cols_to_process:
            print(f"Warning: No score columns found in {eval_name}. Skipping.")
            continue

        # Scale scores and handle potential NaN values
        df[cols_to_process] = scaler.fit_transform(df[cols_to_process])
        df[cols_to_process] = df[cols_to_process].apply(
            lambda row: row.fillna(row.mean()), axis=1
        )

        # Add metadata
        df['Evaluation'] = eval_name
        df['Group'] = assign_group(eval_name)

        # Append the relevant columns to the results list
        final_cols = ['Model'] + UNIFIED_SCORE_COLS + ['Evaluation', 'Group']
        all_results.append(df[final_cols])

    # --- Analysis ---
    if all_results:
        # Combine all individual DataFrames into one
        combined_df = pd.concat(all_results, ignore_index=True)

        # Calculate the sum of scores for each model in each evaluation
        combined_df['Sum_Score'] = combined_df[UNIFIED_SCORE_COLS].sum(axis=1)

        # Group by model, sum the scores across all evaluations, and get the top 5
        best_overall = (combined_df.groupby('Model')['Sum_Score']
                        .sum()
                        .sort_values(ascending=False)
                        .head(5))

        print("\n--- Top 5 Models (Overall) ---")
        print(best_overall)
    else:
        print("\nNo data was processed. Please check file paths and configurations.")
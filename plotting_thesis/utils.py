"""
Shared utilities for plotting scripts.
"""
import pandas as pd
import os
import matplotlib.pyplot as plt


class PlottingConfig:
    """Configuration class for consistent plotting settings."""
    # Standard ROIs and evaluations
    ROIS = ["V1", "V2", "V4", "IT"]
    EVALUATIONS_DICT = {
        'encoding': ["encoding_natural", "encoding_imagenet", "encoding_synthetic", "encoding_illusion"],
        'rsa': ["rsa_natural", "rsa_imagenet", "rsa_synthetic", "rsa_illusion"]
    }
    EVALUATIONS = ["encoding_natural", "encoding_imagenet", "encoding_synthetic", "encoding_illusion",
                   "rsa_natural", "rsa_imagenet", "rsa_synthetic", "rsa_illusion"]
    MAP_DATASET_NAMES_LONG = {
        "rsa_illusion": "Kamitani Illusion",
        "rsa_imagenet": "Kamitani ImageNet",
        "rsa_synthetic": "NSD Synthetic",
        "rsa_natural": "NSD Natural",
        "encoding_illusion": "Kamitani Illusion",
        "encoding_imagenet": "Kamitani ImageNet",
        "encoding_synthetic": "NSD Synthetic",
        "encoding_natural": "NSD Natural",
    }
    MAP_DATASET_NAMES_SHORT = {
        "rsa_illusion": "Illusion",
        "rsa_imagenet": "ImageNet",
        "rsa_synthetic": "Synthetic",
        "rsa_natural": "Natural",
        "encoding_illusion": "Illusion",
        "encoding_imagenet": "ImageNet",
        "encoding_synthetic": "Synthetic",
        "encoding_natural": "Natural",
    }
    MAP_DATASET_TO_EVAL ={
        "rsa_illusion": "rsa",
        "rsa_imagenet": "rsa",
        "rsa_synthetic": "rsa",
        "rsa_natural": "rsa",
        "encoding_illusion": "encoding",
        "encoding_imagenet": "encoding",
        "encoding_synthetic": "encoding",
        "encoding_natural": "encoding"
    }
    # Results directory
    RESULTS_DIR = "../results"
    PLOTS_DIR = "../plots"

    MAP_EVAL_SCORE_NAME ={
        "rsa": "RSA Score R² in %",
        "encoding": "Encoding Score R"
    }
    MAP_EVAL_CAPITALIZE ={
        "rsa": "RSA",
        "encoding": "Encoding"
    }


def load_categories_df():
    return pd.read_csv(PlottingConfig.RESULTS_DIR + "/categories.csv")


def load_accuracies_df():
    return pd.read_csv(PlottingConfig.RESULTS_DIR + "/accuracies.csv")


def load_robustness_df():
    return pd.read_csv(PlottingConfig.RESULTS_DIR + "/effective_robustness.csv")


def load_eval_df(eval):
    return pd.read_csv(PlottingConfig.RESULTS_DIR + f"/{eval}.csv")


def merge_on_model(df1, df2):
    return pd.merge(df1, df2, on="Model", how="inner")


def get_roi_col_name(roi, evaluation):
    return f"%R2_{roi}" if "rsa" in evaluation else f"R_{roi}"


def save_plot(filename, directory=PlottingConfig.PLOTS_DIR):
    os.makedirs(directory, exist_ok=True)
    full_path = os.path.join(directory, filename)
    plt.savefig(full_path)


def filter_df_by_architecture(df, model_type):
    return df[df["architecture"] == model_type]

def filter_df_by_dataset(df, model_type):
    return df[df["dataset"] == model_type]

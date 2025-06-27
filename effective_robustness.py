import pandas as pd
import numpy as np
from scipy.stats import linregress


def get_slope_intercept(id_dataset, ood_dataset):

    accuracies = pd.read_csv("results/accuracies.csv")
    categories = pd.read_csv("results/categories.csv")

    df = pd.merge(accuracies, categories, on='Model', how='inner')
    if ood_dataset == "imagenet-a_1":
        df = df[df["imagenet1k-subset-a"] < 91.86]
        ood_dataset = "imagenet-a"
    elif ood_dataset == "imagenet-a_2":
        df = df[df["imagenet1k-subset-a"] > 91.86]
        ood_dataset = "imagenet-a"

    df = df[df["architecture"] == "CNN"]
    df = df[df["dataset"] == "ImageNet1K"]
    df = df.reset_index(drop=True)

    result = linregress(logit(df[id_dataset]), logit(df[ood_dataset]))

    return result.slope, result.intercept



def logit(acc):
    return np.log(np.divide(acc / 100.0, 1.0 - acc / 100.0))


def inv_logit(acc):
    return (np.exp(acc)/(1 + np.exp(acc)))*100


def effective_robustness(id_accuracy, ood_accuracy, slope, intercept):
    y_pred_logit = logit(id_accuracy) * slope + intercept
    y_pred = inv_logit(y_pred_logit)
    eff_robust = ood_accuracy - y_pred
    return eff_robust


def effective_robustness_csv():
    # first element is intercept, second is slope, third is id imagenet version
    datasets = {"imagenet-r": "imagenet1k-subset-r",
                "imagenet-sketch": "imagenet1k",
                "imagenetv2-matched-frequency": "imagenet1k",
                "imagenet-a_1": "imagenet1k-subset-a",
                "imagenet-a_2": "imagenet1k-subset-a"}
    """    line_fit = {"imagenet-r": [-2.1077156197680713,  0.636843984281646, "imagenet1k-subset-r"],
                    "imagenet-sketch": [-2.370072912552283, 1.0709154135668684, "imagenet1k"],
                    "imagenetv2-matched-frequency": [-0.4813069457734013, 0.9113725552359271, "imagenet1k"],
                    "imagenet-a_1": [-5.14004617225444,   0.6894464990989362, "imagenet1k-subset-a"],
                    "imagenet-a_2": [-10.27739708860645,  2.9559595169134285, "imagenet1k-subset-a"]}"""

    acc = pd.read_csv("results/accuracies.csv")
    for col in acc.columns:
        if col == "imagenet-a":
            acc[col] = acc.apply(
                lambda row: effective_robustness(
                    row[datasets["imagenet-a_1"]],
                    row[col],
                    # resnet50 has 91.86 id accuracy
                    get_slope_intercept(datasets["imagenet-a_1"], "imagenet-a_1")[0] if row[datasets["imagenet-a_1"]] < 91.86 else  get_slope_intercept(datasets["imagenet-a_2"], "imagenet-a_2")[0],
                    get_slope_intercept(datasets["imagenet-a_1"], "imagenet-a_1")[1] if row[datasets["imagenet-a_1"]] < 91.86 else  get_slope_intercept(datasets["imagenet-a_2"], "imagenet-a_2")[1]
                ),
                axis=1
            )
        elif col not in ["imagenet1k", "Model", "imagenet1k-subset-r", "imagenet1k-subset-a"]:
            acc[col] = acc.apply(lambda row: effective_robustness(row[datasets[col]], row[col], get_slope_intercept(datasets[col], col)[0], get_slope_intercept(datasets[col], col)[1]), axis=1)
    acc.drop(columns=["imagenet1k","imagenet1k-subset-r", "imagenet1k-subset-a"], inplace=True)
    csv_filename = 'results/effective_robustness.csv'
    acc.to_csv(csv_filename, mode='w', index=False, header=True)


def effective_brain_similarity_csv():
    line_fit = {
        "encoding_synthetic": {
            "R_V4": [-0.33814694199631284, -0.1476222150447993],
            "R_IT": [-1.0639758030202169, -0.028793081082354326],
            "R_V1": [0.24887632015535982, 0.0611023893261019],
            "R_V2": [-0.010968508966649781, 0.017796523270572958]
        },
        "rsa_synthetic": {
            "%R2_V4": [0.27392396596556556, 0.24692392181878406],
            "%R2_IT": [0.07613594450436376, 0.19168615725751662],
            "%R2_V1": [0.6611876350997855, 0.549548460473435],
            "%R2_V2": [0.2939274052370374, 0.5352755795914498]
        },
        "encoding_illusion": {
            "R_V4": [-0.4349823361293494, -0.028549062874587233],
            "R_IT": [-1.2655756323643699, -0.011149690093234473],
            "R_V1": [-0.1853297044536728, 0.06013989586963853],
            "R_V2": [-0.16548621910346117, 0.05818063641625954]
        },
        "rsa_illusion": {
            "%R2_V4": [0.8749587619621217, 0.38067354865269454],
            "%R2_IT": [1.1658316116599814, 0.3596580797668323],
            "%R2_V1": [0.8068001062848853, 0.2045143806286372],
            "%R2_V2": [0.7664999685216558, 0.3960703099861144]
        }
    }

    df = pd.DataFrame()

    for i, key in enumerate(line_fit):
        base_name = key.split('_')[0]
        id_df = pd.read_csv(f"results/{base_name}.csv")
        ood_df = pd.read_csv(f"results/{key}.csv")
        df_merged = pd.merge(id_df, ood_df, on='Model', how='inner')
        for roi in id_df.columns:
            if roi == "Model":
                df["Model"] = df_merged["Model"]
            elif "encoding" in key:
                intercept, slope = line_fit[key][roi]
                df[roi+f"_{key}"] = df_merged.apply(lambda row: effective_robustness(row[roi+"_x"]*100, row[roi+"_y"]*100, intercept, slope), axis=1)
            else:
                intercept, slope = line_fit[key][roi]
                df[roi+f"_{key}"] = df_merged.apply(lambda row: effective_robustness(row[roi+"_x"], row[roi+"_y"], intercept, slope), axis=1)

    df.to_csv("results/effective_brain_similarity.csv", index=False, header=True)


if __name__ == '__main__':
    effective_robustness_csv()
    effective_brain_similarity_csv()


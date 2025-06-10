import pandas as pd
import numpy as np


def logit(acc):
    return np.log(np.divide(acc / 100.0, 1.0 - acc / 100.0))


def inv_logit(acc):
    return (np.exp(acc)/(1 + np.exp(acc)))*100


def effective_robustness(id_accuracy, ood_accuracy, intercept, slope):
    y_pred_logit = logit(id_accuracy) * slope + intercept
    y_pred = inv_logit(y_pred_logit)
    eff_robust = ood_accuracy - y_pred
    return eff_robust


def effective_robustness_csv():
    # first element is intercept, second is slope, third is id imagenet version
    line_fit = {"imagenet-r": [-2.1077156197680713,  0.636843984281646, "imagenet1k-subset-r"],
                "imagenet-sketch": [-2.370072912552283, 1.0709154135668684, "imagenet1k"],
                "imagenetv2-matched-frequency": [-0.4813069457734013, 0.9113725552359271, "imagenet1k"],
                "imagenet-a_1": [-5.14004617225444,   0.6894464990989362, "imagenet1k-subset-a"],
                "imagenet-a_2": [-10.27739708860645,  2.9559595169134285, "imagenet1k-subset-a"]}

    acc = pd.read_csv("results/accuracies.csv")
    for col in acc.columns:
        if col == "imagenet-a":
            acc[col] = acc.apply(
                lambda row: effective_robustness(
                    row[line_fit["imagenet-a_1"][2]],
                    row[col],
                    line_fit["imagenet-a_1"][0] if row["imagenet1k-subset-a"] < 91.86 else line_fit["imagenet-a_2"][0],
                    line_fit["imagenet-a_1"][1] if row["imagenet1k-subset-a"] < 91.86 else line_fit["imagenet-a_2"][1]
                ),
                axis=1
            )
        elif col not in ["imagenet1k", "Model", "imagenet1k-subset-r", "imagenet1k-subset-a"]:
            acc[col] = acc.apply(lambda row: effective_robustness(row[line_fit[col][2]], row[col], line_fit[col][0], line_fit[col][1]), axis=1)
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
        }
    }

    df = pd.DataFrame()

    for i, key in enumerate(line_fit):
        base_name = key.split('_')[0]
        id_df = pd.read_csv(f"results/{base_name}.csv")
        ood_df = pd.read_csv(f"results/{key}.csv")

        # Add the "Model" column once from the first file
        if i == 0:
            df["Model"] = ood_df["Model"]

        for col in ood_df.columns:
            if col == "Model":
                continue
            intercept, slope = line_fit[key][col]
            if "encoding" in key:
                df[f"{col}_{key}"] = [
                    effective_robustness(id_df.loc[idx, col]*100,
                                         ood_df.loc[idx, col]*100,
                                         slope, intercept)
                    for idx in range(len(ood_df))
                ]
            else:
                df[f"{col}_{key}"] = [
                    effective_robustness(id_df.loc[idx, col],
                                         ood_df.loc[idx, col],
                                         slope, intercept)
                    for idx in range(len(ood_df))
                ]

    df.to_csv("results/effective_brain_similarity.csv", index=False, header=True)


if __name__ == '__main__':
    effective_robustness_csv()
    effective_brain_similarity_csv()


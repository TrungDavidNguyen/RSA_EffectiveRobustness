import os

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np


def create_plot(ood_dataset, roi_id, roi_ood, evaluation, evaluation_ood):
    roi_name_id = f"%R2_{roi_id}" if "rsa" in evaluation else f"R_{roi_id}"
    roi_name_ood = f"%R2_{roi_ood}" if "rsa" in evaluation_ood else f"R_{roi_ood}"

    brain_similarity = pd.read_csv(f"../results/{evaluation}.csv")
    brain_similarity_ood = pd.read_csv(f"../results/{evaluation_ood}.csv")

    robustness = pd.read_csv("../results/effective_robustness.csv")
    categories = pd.read_csv("../results/categories.csv")

    brain_similarity = brain_similarity.dropna(subset=[roi_name_id])
    brain_similarity_ood = brain_similarity_ood.dropna(subset=[roi_name_ood])

    brain_similarity = brain_similarity[['Model', roi_name_id]].rename(columns={roi_name_id: f"{roi_name_id}_id"})
    brain_similarity_ood = brain_similarity_ood[['Model', roi_name_ood]].rename(columns={roi_name_ood: f"{roi_name_ood}_ood"})

    df = pd.merge(brain_similarity, robustness, on='Model', how='inner')
    df = pd.merge(df, brain_similarity_ood, on='Model', how='inner')
    df = pd.merge(df, categories, on='Model', how='inner')

    if ood_dataset == "imagenet-a":
        df = df[df['Model'].str.lower() != "resnet50"]
        df = df.reset_index(drop=True)
    df = df[df["architecture"] == "CNN"]

    architectures = df['architecture'].unique()
    colors = plt.cm.tab10.colors
    color_map = {arch: colors[i % len(colors)] for i, arch in enumerate(architectures)}
    df['color'] = df['architecture'].map(color_map)

    print(df.columns)
    X = df[[f"{roi_name_id}_id", f"{roi_name_ood}_ood"]].values
    Z = df[ood_dataset].values
    X_with_const = sm.add_constant(X)
    model = sm.OLS(Z, X_with_const).fit()

    c, a, b = model.params

    x_surf, y_surf = np.meshgrid(
        np.linspace(df[f"{roi_name_id}_id"].min(), df[f"{roi_name_id}_id"].max(), 10),
        np.linspace(df[f"{roi_name_ood}_ood"].min(), df[f"{roi_name_ood}_ood"].max(), 10)
    )
    z_surf = a * x_surf + b * y_surf + c

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[f"{roi_name_id}_id"], df[f"{roi_name_ood}_ood"], Z, c=df['color'], s=50)

    ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.3, color='gray', edgecolor='none')

    ax.set_xlabel(evaluation + " " + roi_name_id)
    ax.set_ylabel(evaluation_ood + " " + roi_name_ood)
    ax.set_zlabel(ood_dataset + " Effective Robustness")
    plt.title(f"{roi_name_id} vs {roi_name_ood}")
    os.makedirs(f"../plots/3d", exist_ok=True)
    plt.savefig(f"../plots/3d/{ood_dataset}_{evaluation}_{evaluation_ood}_{roi_name_id}_{roi_name_ood}.png")

    for arch in architectures:
        ax.scatter([], [], [], color=color_map[arch], label=arch)
    ax.legend(title="Architecture")
    print(model.summary())

    plt.show()




if __name__ == '__main__':
    create_plot("imagenet-r","IT", "IT","rsa_synthetic", "rsa_illusion")
    #create_plot("imagenet-r","V2", "V4","rsa_natural", "rsa_synthetic")
    #create_plot("imagenet-r","V2", "V1","rsa_imagenet", "rsa_illusion")

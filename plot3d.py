import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np

def create_plot(ood_dataset, roi, evaluation, evaluation_ood):
    eval_name = "%R2" if evaluation in ["rsa", "rsa_synthetic"] else "R"
    roi_name = f"%R2_{roi}" if evaluation in ["rsa", "rsa_synthetic"] else f"R_{roi}"

    brain_similarity = pd.read_csv(f"results/{evaluation}.csv")
    brain_similarity_ood = pd.read_csv(f"results/{evaluation_ood}.csv")
    brain_similarity_ood = brain_similarity_ood.rename(columns={roi_name: f"{roi_name}_ood"})

    robustness = pd.read_csv("results/effective_robustness.csv")
    categories = pd.read_csv("results/categories.csv")

    # Drop missing values
    brain_similarity = brain_similarity.dropna(subset=[roi_name])
    brain_similarity_ood = brain_similarity_ood.dropna(subset=[f"{roi_name}_ood"])

    # Merge data
    df = pd.merge(brain_similarity, robustness, on='Model', how='inner')
    df = pd.merge(df, brain_similarity_ood, on='Model', how='inner')
    df = pd.merge(df, categories, on='Model', how='inner')

    if ood_dataset == "imagenet-a":
        df = df[df['Model'].str.lower() != "resnet50"]
        df = df.reset_index(drop=True)
    #df = df[df["dataset"] != "more data"]
    #df = df[df["architecture"] != "VIT"]
    # Color map
    architectures = df['architecture'].unique()
    colors = plt.cm.tab10.colors
    color_map = {arch: colors[i % len(colors)] for i, arch in enumerate(architectures)}
    df['color'] = df['architecture'].map(color_map)

    # Get X, Y, Z for plane fitting
    X = df[[roi_name, f"{roi_name}_ood"]].values
    Z = df[ood_dataset].values
    X_with_const = sm.add_constant(X)  # Adds intercept term
    model = sm.OLS(Z, X_with_const).fit()

    # Coefficients
    c, a, b = model.params  # Intercept, coef_x, coef_y

    # Plane surface
    x_surf, y_surf = np.meshgrid(
        np.linspace(df[roi_name].min(), df[roi_name].max(), 10),
        np.linspace(df[f"{roi_name}_ood"].min(), df[f"{roi_name}_ood"].max(), 10)
    )
    z_surf = a * x_surf + b * y_surf + c


    # 3D Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[roi_name], df[f"{roi_name}_ood"], Z, c=df['color'], s=50)

    # Plot plane
    ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.3, color='gray', edgecolor='none')

    ax.set_xlabel(evaluation + " " + eval_name)
    ax.set_ylabel(evaluation_ood + " " + eval_name)
    ax.set_zlabel(ood_dataset + " Effective Robustness")
    plt.title(roi)

    # Legend
    for arch in architectures:
        ax.scatter([], [], [], color=color_map[arch], label=arch)
    ax.legend(title="Architecture")

    plt.show()

    # Print regression stats
    print(model.summary())



if __name__ == '__main__':
    for ood_dataset in ["imagenet-r","imagenet-sketch", "imagenetv2-matched-frequency","imagenet-a"]:
        for roi in ["V1", "V2", "V4","IT"]:
            create_plot(ood_dataset, roi, "rsa","rsa_synthetic")
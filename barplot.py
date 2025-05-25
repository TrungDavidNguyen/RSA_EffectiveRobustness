import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


def create_bar_plot(method, rois, model=None):
    df = pd.read_csv(f"results/{method}.csv")
    if method in ["rsa", "rsa_synthetic"]:
        eval = "%R2"
    else:
        eval = "R"

    # Set model as index
    df_plot = df.set_index('Model')
    order = [f"{eval}_{roi}" for roi in rois]
    df_plot = df_plot[order]

    if model is not None:
        df_plot = df_plot.loc[[model]]

    fig, ax = plt.subplots(figsize=(16, 6))  # Wider layout

    df_plot.plot(kind='bar', ax=ax)

    ax.set_xlabel('Model')
    ax.set_ylabel(method)
    ax.set_title(f'{method} {eval} results per Model')
    ax.legend(title=f'{method}/ROI')

    plt.tight_layout()
    plt.savefig(f"plots/barplot_{method}.png")
    plt.show()


def corr(method, rois):
    nsd = pd.read_csv(f"results/{method}.csv")
    nsd_synthetic = pd.read_csv(f"results/{method}_synthetic.csv")

    df = pd.merge(nsd, nsd_synthetic, on='Model', how='inner', suffixes=(f' {method}', f' {method}_synthetic'))

    if method =="rsa":
        eval = "%R2"
    elif method == "encoding":
        eval = "R"
    df = df.dropna(subset=[f"{eval}_{rois[0]} {method}"])
    #df = df.head()
    slope, intercept, r_value, p_value, std_err = linregress(df[f"{eval}_{rois[0]} {method}_synthetic"], df[f"{eval}_{rois[0]} {method}"])
    print(r_value)

if __name__ == '__main__':
    create_bar_plot("encoding_synthetic", ["V1","V2","V4","IT"])

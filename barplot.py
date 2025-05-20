import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


def create_bar_plot(method, rois, model=None):
    nsd = pd.read_csv(f"results/{method}.csv")
    nsd_synthetic = pd.read_csv(f"results/{method}_synthetic.csv")

    df = pd.merge(nsd, nsd_synthetic, on='Model', how='inner', suffixes=(f' {method}', f' {method}_synthetic'))

    if method =="rsa":
        eval = "%R2"
    elif method == "encoding":
        eval = "R"
    df = df.dropna(subset=[f"{eval}_{rois[0]} {method}"])

    slope, intercept, r_value, p_value, std_err = linregress(df[f"{eval}_{rois[0]} {method}_synthetic"], df[f"{eval}_{rois[0]} {method}"])
    print(r_value)
    # Set model as index
    df_plot = df.set_index('Model')
    order = []
    for dataset in [method, f"{method}_synthetic"]:
        for roi in rois:
            order.append(f"{eval}_{roi} {dataset}")
    df_plot = df_plot[order]

    # Plot
    if model is not None:
        df_plot = df_plot.loc[[model]]
    df_plot.plot(kind='bar')

    # Add labels
    plt.xlabel('Model')
    plt.ylabel(method)
    plt.title(f'{method} results per Model')
    plt.legend(title='Category')  # Legend for 'a' and 'b'
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    create_bar_plot("encoding", ["V1"])

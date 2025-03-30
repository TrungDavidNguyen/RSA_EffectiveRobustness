import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


def create_plot():
    df = pd.read_csv("results/results-imagenet-sketch-IT.csv")
    #df = df[~df['Model'].str.contains("Densenet", case=False, na=False)]
    df = df.reset_index()
    # create scatter plot with model names
    plt.scatter(df["%R2"], df["eff.Robustness"], marker="o", color="blue")

    for i in range(len(df)):
        plt.text(df.loc[i, "%R2"], df.loc[i, "eff.Robustness"], df.loc[i, "Model"],
                 fontsize=7, ha='right', va='bottom')

    # fit line
    slope, intercept, r_value, p_value, std_err = linregress(df["%R2"], df["eff.Robustness"])
    x_vals = df["%R2"]
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color="red")
    # add correlation
    plt.text(min(df["%R2"]), max(df["eff.Robustness"]), f"r = {r_value:.2f}")

    plt.xlabel("Brain Similarity")
    plt.ylabel("Effective Robustness")
    plt.title("IT and imagenet-sketch")
    plt.savefig("plots/IT_imagenet-sketch")
    plt.show()


if __name__ == '__main__':
    create_plot()
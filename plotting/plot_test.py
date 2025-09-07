import pandas as pd
from net2brain.evaluations.plotting import Plotting

# Replace 'your_file.csv' with the path to your CSV file
model = "GoogleNet"
dataset = "rsa_illusion"
df = pd.read_csv(fr"C:\Users\david\Desktop\RSA_EffectiveRobustness\{dataset}\V1\rsa_V1_mean\{model}_RSA.csv")
df_1 = pd.read_csv(fr"C:\Users\david\Desktop\RSA_EffectiveRobustness\{dataset}\V2\rsa_V2_mean\{model}_RSA.csv")
df_2 = pd.read_csv(fr"C:\Users\david\Desktop\RSA_EffectiveRobustness\{dataset}\V4\rsa_V4_mean\{model}_RSA.csv")
df_3 = pd.read_csv(fr"C:\Users\david\Desktop\RSA_EffectiveRobustness\{dataset}\IT\rsa_IT_mean\{model}_RSA.csv")

single_plotter = Plotting([df, df_1, df_2, df_3])
single_plotter.plot_all_layers(metric='R2')


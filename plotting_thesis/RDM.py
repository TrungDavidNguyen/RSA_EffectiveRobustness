import numpy as np
import matplotlib.pyplot as plt
import os

# Load your RDM (assuming it's a square matrix)
rdm = np.load(r"..\NSD Dataset/NSD_872_RDMs/prf-visualrois/combined/V4_both_fmri.npz")["rdm"][0][:3, :3]
plt.imshow(rdm, cmap="viridis", interpolation="nearest")
plt.colorbar(label="Dissimilarity")
plt.title("Random Dissimilarity Matrix")

output_dir = f"../plots/thesis"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(f"{output_dir}/RDM.png")

plt.show()

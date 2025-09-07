import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression

np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

model = LinearRegression()
model.fit(X, y)

X_new = np.linspace(0, 2, 50).reshape(50, 1)
y_predict = model.predict(X_new)

plt.figure(figsize=(4, 3))
plt.scatter(X, y, label="Data points")
plt.plot(X_new, y_predict, "r-", linewidth=2, label="Linear regression line")

# Bigger and bold title and labels
plt.title("Encoding Model", fontsize=16, fontweight="bold")
plt.xlabel("Model Features", fontsize=14, fontweight="bold")
plt.ylabel("Neural Data", fontsize=14, fontweight="bold")

plt.xticks([])
plt.yticks([])

output_dir = f"../plots/thesis"
os.makedirs(output_dir, exist_ok=True)
plt.savefig(f"{output_dir}/lin_reg.png")
plt.show()

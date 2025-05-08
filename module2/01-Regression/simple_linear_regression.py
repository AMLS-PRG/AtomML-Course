"""
Simple Linear Regression Demo
-----------------------------
This script demonstrates fitting a simple linear regression model to synthetic data.
We simulate house size vs price (1 feature).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(0)
X = np.random.uniform(50, 150, 100).reshape(-1, 1)  # House size in square meters
noise = np.random.normal(0, 8, 100).reshape(-1, 1)
y = 3 * X + 20 + noise  # Linear relationship with noise

# Fit simple linear regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)

# Plot results
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="blue", label="True Price", alpha=0.6)
plt.plot(X, y_pred, color="red", label="Predicted Line")
plt.title(f"Simple Linear Regression\nMSE = {mse:.2f}")
plt.xlabel("House Size (㎡)")
plt.ylabel("Price (×10,000 RMB)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Detect outliers using "Ozone" and "Temp".
import pandas as pd
import numpy as np
from scipy.stats import chi2
from matplotlib import patches
import matplotlib.pyplot as plt

# Imports dataset and clean it
df = pd.read_csv("airquality.csv", sep=",", decimal=".")
df.head()

df = df[["Ozone","Solar.R","Wind","Temp"]]
df = df.dropna()
df = df.to_numpy()
print(f"\n\nDF: {df}\n\n")
# 1. Calculate the covariance matrix
cov = np.cov(df, rowvar=False)

# Covariance matrix power of -1
# Covariance matrix indicates how variables variate together
covariance_pm1 = np.linalg.matrix_power(cov, -1)
print(f"Covariance matrix: {covariance_pm1}")

# 2. Center point
centerpoint = np.mean(df, axis=0)
print(f"Center point of Ozone and Temp: {centerpoint}")

# 3. Find the distance between the center point and each observation point in the dataset.
# We need to find the cutoff value from the Chi-Square distribution.
distances = []
for i, val in enumerate(df):
    p1 = val  # Ozone and Temp of the ith row
    p2 = centerpoint
    distance = (p1 - p2).T.dot(covariance_pm1).dot(p1 - p2)
    distances.append(distance)
    # print(f"Distance: {distance}")
distances = np.array(distances)

# 4. Cutoff (threshold) value from Chi-Square Distribution for detecting outliers
cutoff = chi2.ppf(0.95, df.shape[1])

# Index of outliers
outlierIndexes = np.where(distances > cutoff)

print("--- Index of Outliers ----")
print(outlierIndexes)
# array([24, 35, 67, 81])

print("--- Observations found as outlier -----")
print(df[distances > cutoff, :])
# [[115.  79.], [135.  84.], [122.  89.], [168.  81.]]

## 5. Finding ellipse dimensions
pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
ell_radius_x = np.sqrt(1 + pearson)
ell_radius_y = np.sqrt(1 - pearson)
lambda_, v = np.linalg.eig(cov)
lambda_ = np.sqrt(lambda_)

# Ellipse patch
ellipse = patches.Ellipse(
    xy=(centerpoint[0], centerpoint[1]),
    width=lambda_[0] * np.sqrt(cutoff) * 2,
    height=lambda_[1] * np.sqrt(cutoff) * 2,
    angle=np.rad2deg(np.arccos(v[0, 0])),
    edgecolor="#fab1a0",
)
ellipse.set_facecolor("#0984e3")
ellipse.set_alpha(0.5)
fig = plt.figure()
ax = plt.subplot()
ax.add_artist(ellipse)
plt.scatter(df[:, 0], df[:, 1])
plt.show()

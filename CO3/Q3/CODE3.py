import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --------------------------------------------------
# 1. Load Iris Dataset
# --------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

print("Original Dataset Shape:")
print(X.shape)

print("\nFeatures:")
for feature in feature_names:
    print(feature)

# --------------------------------------------------
# 2. Standardize the data
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# 3. Visualize BEFORE PCA
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=y
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Features Before PCA")

plt.grid(True)
plt.show()

# --------------------------------------------------
# 4. Apply PCA
# --------------------------------------------------

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

# --------------------------------------------------
# 5. Display PCA information
# --------------------------------------------------

print("\nPCA Dataset Shape:")
print(X_pca.shape)

print("\nExplained Variance Ratio:")

for i, variance in enumerate(pca.explained_variance_ratio_):
    print(
        f"PC{i + 1}: {variance * 100:.2f}%"
    )

total_variance = sum(
    pca.explained_variance_ratio_
)

print(
    f"\nTotal Variance Preserved: "
    f"{total_variance * 100:.2f}%"
)

# --------------------------------------------------
# 6. Visualize AFTER PCA
# --------------------------------------------------

plt.figure(figsize=(8, 6))

for i in range(3):

    plt.scatter(
        X_pca[y == i, 0],
        X_pca[y == i, 1],
        label=target_names[i]
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("Features After PCA")

plt.legend()
plt.grid(True)

plt.show()

# --------------------------------------------------
# 7. Compare original and reduced dimensions
# --------------------------------------------------

print("\n===================================")
print("        PCA DIMENSIONALITY")
print("===================================")

print("Original dimensions :", X.shape[1])
print("Reduced dimensions  :", X_pca.shape[1])

print(
    "Information preserved:",
    f"{total_variance * 100:.2f}%"
)

# --------------------------------------------------
# 8. Show first 10 transformed samples
# --------------------------------------------------

print("\nFirst 10 PCA Transformed Samples:")

print(
    f"{'PC1':>12}"
    f"{'PC2':>12}"
)

print("-" * 25)

for i in range(10):

    print(
        f"{X_pca[i, 0]:>12.4f}"
        f"{X_pca[i, 1]:>12.4f}"
    )

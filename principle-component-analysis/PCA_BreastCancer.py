# Principle Component Analysis (PCA) - Wisconsin Breast Cancer Dataset
# Proof-of-Concept for dimensionality reduction

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time

# Start timer
t0 = time.time()  # Add at start of process

# 1. Load the dataset
data = load_breast_cancer()
feature_names = data.feature_names
df = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 0: Malignant, 1: Benign

# 2. Standardize the data (Mean=0, Variance=1)
# This is vital because PCA is sensitive to the scale of the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# 3. Apply PCA
# We'll reduce the 30 features down to 2 principal components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

# 4. Visualize the results
plt.figure(figsize=(8, 6))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA: Breast Cancer Wisconsin Dataset')
plt.legend(handles=scatter.legend_elements()[0], labels=['Malignant', 'Benign'])
plt.grid(True)
plt.tight_layout()
plt.savefig('pca_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Check Explained Variance
print(f"Variance explained by first two components: {sum(pca.explained_variance_ratio_):.2%}")

# 6. Visualise heatmap of components
comp = pd.DataFrame(pca.components_, columns=feature_names)
plt.figure(figsize=(8, 6))
sns.heatmap(comp, cmap='viridis')
plt.ylabel('Principal Component ID')
plt.title('PCA: 2 Components Feature Heatmap')
plt.tight_layout()
plt.savefig('pca_comp_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Track time to complete process
t1 = time.time()  # Add at end of process
timetaken1 = t1 - t0
print(f"\nTime Taken: {timetaken1:.4f} seconds")
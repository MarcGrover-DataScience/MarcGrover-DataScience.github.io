# Principle Component Analysis (PCA) - Wisconsin Breast Cancer Dataset
# Proof-of-Concept for dimensionality reduction

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import time

# Start timer
t0 = time.time()  # Add at start of process

# 1. Load the dataset
data = load_breast_cancer()
feature_names = data.feature_names
df = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 0: Malignant, 1: Benign

# 1.2. Dataset Validation
print("Dataset Validation")
print(f"Shape: {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum().sum()} total missing values")
print(f"\nClass distribution:")
unique, counts = np.unique(y, return_counts=True)
labels = data.target_names  # ['malignant', 'benign']
for label, count in zip(labels, counts):
    print(f"  {label.capitalize()}: {count} ({count/len(y):.1%})")
print(f"\nDescriptive Statistics (first 5 features):")
print(df.iloc[:, :5].describe().round(3))

# 2. Standardize the data (Mean=0, Variance=1)
# This is vital because PCA is sensitive to the scale of the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# 2.1. Correlation Matrix (PCA Justification)
plt.figure(figsize=(14, 12))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Upper triangle mask
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm',
            center=0, vmin=-1, vmax=1,
            linewidths=0.3, linecolor='white')
plt.title('Feature Correlation Matrix: Wisconsin Breast Cancer Dataset', fontsize=14)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Print the strongly correlated pairs
print("\nHighly Correlated Feature Pairs (|r| > 0.90)")
corr_pairs = corr_matrix.abs().unstack()
corr_pairs = corr_pairs[corr_pairs < 1.0].drop_duplicates().sort_values(ascending=False)
print(corr_pairs[corr_pairs > 0.90].to_string())


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

# 6. Visualise the loadings (weights) as heatmap

# 6.1. Extract the loadings (weights) PC1 and PC2
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=data.feature_names
)

# 6.2. Plotting the Heatmap for the features
plt.figure(figsize=(10, 6))
sns.heatmap(loadings, annot=True, cmap='viridis')
plt.title('Feature Loadings for PC1 and PC2')
plt.tight_layout()
plt.savefig('pca_comp_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 6.3. Print top loadings for PC1 and PC2
print("\n=== Top Feature Loadings ===")
for pc in ['PC1', 'PC2']:
    print(f"\n{pc} — Top 5 features (by absolute loading):")
    top = loadings[pc].abs().sort_values(ascending=False).head(5)
    for feat in top.index:
        val = loadings.loc[feat, pc]
        print(f"  {feat:<35} {val:+.4f}")

# 7. Biplot - Understanding the PCA Loadings

# Define a function to plot the Biplot chart
def plot_pca_biplot(score, coeff, labels=None, targets=None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]

    # Scale the scores to fit the -1 to 1 range of the vectors
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())

    plt.figure(figsize=(10, 7))

    # Plot the data points
    scatter = plt.scatter(xs * scalex, ys * scaley, c=targets, cmap='viridis', alpha=0.3)

    # Plot the arrows (vectors)
    # Only plotting the first 10 features for clarity; 30 is too crowded!
    for i in range(min(n, 10)):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.8, head_width=0.02)
        if labels is None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, "Var" + str(i + 1), color='black', ha='center',
                     va='center')
        else:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='black', ha='center', va='center')

    plt.legend(handles=scatter.legend_elements()[0],
               labels=['Malignant', 'Benign'],
               loc='upper left', title='Diagnosis')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Biplot (First 10 Features)")
    plt.grid(True)


# Execute the function
plot_pca_biplot(pca_result, np.transpose(pca.components_), labels=data.feature_names, targets=y)
plt.tight_layout()
plt.savefig('pca_biplot.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. Understanding variance accounted for by PCA components

# 8.0. Fit PCA without reducing components first to see the full picture
pca2 = PCA()
pca2.fit(scaled_data)

# 8.1. Calculate Variance Metrics
exp_var_ratio = pca2.explained_variance_ratio_
cum_var_ratio = np.cumsum(exp_var_ratio)

# 8.2. Identify the 90% Threshold
n_components_90 = np.argmax(cum_var_ratio >= 0.90) + 1

# 8.3. Visualise the Scree Plot
plt.figure(figsize=(10, 6))

# Bar chart for individual variance
plt.bar(range(1, len(exp_var_ratio) + 1), exp_var_ratio, alpha=0.5,
        align='center', label='Individual Explained Variance', color='skyblue')

# Step plot for cumulative variance
plt.step(range(1, len(cum_var_ratio) + 1), cum_var_ratio, where='mid',
         label='Cumulative Explained Variance', color='navy', lw=2)

# Annotation for the 90% mark
plt.axhline(y=0.9, color='red', linestyle='--', alpha=0.6, label='90% Threshold')
plt.axvline(x=n_components_90, color='green', linestyle=':', label=f'90% at {n_components_90} PCs')
elbow = 3
plt.annotate(f'Elbow ≈ PC{elbow}\n({cum_var_ratio[elbow-1]:.1%})',
             xy=(elbow, cum_var_ratio[elbow-1]),
             xytext=(elbow + 1.5, cum_var_ratio[elbow-1] - 0.08),
             arrowprops=dict(arrowstyle='->', color='darkorange'),
             color='darkorange', fontsize=9)

plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Component Index')
plt.title('Scree Plot: Identifying the "Elbow" & Variance Threshold')
plt.legend(loc='best')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('pca_variance_analysis.png')
plt.show()

# 9. PC1 vs PC3 Scatter (elbow component)
pca3 = PCA(n_components=3)
pca3_result = pca3.fit_transform(scaled_data)

plt.figure(figsize=(8, 6))
scatter3 = plt.scatter(pca3_result[:, 0], pca3_result[:, 2],
                       c=y, cmap='viridis', alpha=0.7)
plt.xlabel(f'PC1 ({pca3.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC3 ({pca3.explained_variance_ratio_[2]:.1%} variance)')
plt.title('PCA: PC1 vs PC3 — Elbow Component')
plt.legend(handles=scatter3.legend_elements()[0], labels=['Malignant', 'Benign'])
plt.grid(True)
plt.tight_layout()
plt.savefig('pca_scatter_pc1_pc3.png', dpi=300, bbox_inches='tight')
plt.show()

# Print the report
print(f"--- Variance Report ---")
for i, var in enumerate(exp_var_ratio[:10]): # Showing first 10 for brevity
    print(f"PC{i+1}: {var:.2%} (Cumulative: {cum_var_ratio[i]:.2%})")
print(f"\nTotal components needed for 90% variance: {n_components_90}")

# Track time to complete process
t1 = time.time()  # Add at end of process
timetaken1 = t1 - t0
print(f"\nTime Taken: {timetaken1:.4f} seconds")
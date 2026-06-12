import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, silhouette_samples, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             homogeneity_score, completeness_score, v_measure_score)
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import warnings

warnings.filterwarnings('ignore')

# Dataframe presentation configuration
desired_width = 320                                                 # shows columns with X or fewer characters
pd.set_option("display.width", desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 10)                            # shows Y columns in the display
pd.set_option("display.max_rows", 20)                               # shows Z rows in the display
pd.set_option("display.min_rows", 10)                               # defines the minimum number of rows to show
pd.set_option("display.precision", 3)                               # displays numbers to 3 dps

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================
print("STEP 1: Loading and Exploring Data")

# Load the dataset (adjust path as needed)
# The seed dataset has 7 features and 1 target column
df = pd.read_csv('Seed_Data.csv')

print(f"\nDataset shape: {df.shape}")
print("Data sample")
print(df)
print(f"\nDataset info:\n")
print(df.info())
print(f"\nBasic statistics:\n{df.describe()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Separate features and target
# Assuming the last column is the target (variety)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Unique classes in target: {np.unique(y)}")

# ============================================================================
# 1b. FEATURE DISTRIBUTIONS & SKEWNESS CHECK
# ============================================================================
print("\nSTEP 1b: Feature Distributions and Skewness")

feature_names = df.columns[:-1].tolist()   # all columns except the last (target)

skewness = df[feature_names].skew().round(3)
print("\nFeature Skewness (|skew| > 1.0 is materially skewed):")
print(skewness)

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
for i, col in enumerate(feature_names):
    axes[i].hist(df[col], bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    axes[i].set_title(f'{col}\nskew = {skewness[col]:.2f}', fontsize=10)
axes[-1].set_visible(False)
plt.suptitle('Feature Distributions (Raw Data)', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('kmeans-feat-distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 1c. FEATURE CORRELATION HEATMAP
# ============================================================================
print("\nSTEP 1c: Feature Correlation Analysis")

corr = df[feature_names].corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, ax=ax, annot_kws={'size': 9})
ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('kmeans-correlation.png', dpi=150, bbox_inches='tight')
plt.show()

# Print highly correlated pairs for reporting
print("\nHighly correlated feature pairs (|r| > 0.85):")
for i in range(len(corr.columns)):
    for j in range(i):
        if abs(corr.iloc[i, j]) > 0.85:
            print(f"  {corr.columns[i]} vs {corr.columns[j]}: r = {corr.iloc[i, j]:.3f}")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\nSTEP 2: Data Preprocessing (Standardization)")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nScaled data - Mean: {X_scaled.mean(axis=0)}")
print(f"Scaled data - Std: {X_scaled.std(axis=0)}")

# ============================================================================
# 3. ELBOW METHOD - FINDING OPTIMAL K
# ============================================================================
print("\nSTEP 3: Elbow Method - Finding Optimal K")

# Calculate WSS (Within-Cluster Sum of Squares) for different k values
k_range = range(2, 11)
wss = []
silhouette_scores = []
davies_bouldin_scores = []
calinski_harabasz_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wss.append(kmeans.inertia_)

    # Calculate intrinsic metrics
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels))
    calinski_harabasz_scores.append(calinski_harabasz_score(X_scaled, labels))

# print("\nWSS values for different k:")
# for k, wss_val in zip(k_range, wss):
#     print(f"k={k}: WSS={wss_val:.2f}")

# Dataframe of WSS values for Elbow method
elbow_data = pd.DataFrame({
    'Number of Clusters (k)': list(k_range),
    'WSS': wss
})
print('\nTable of WSS scores, to be used for elbow method')
print(elbow_data)

# DataFrame for Silhouette Scores
silhouette_data = pd.DataFrame({
    'Number of Clusters (k)': list(k_range),
    'Silhouette Score': silhouette_scores
})

# DataFrame for Davies-Bouldin Index
dbi_data = pd.DataFrame({
    'Number of Clusters (k)': list(k_range),
    'Davies-Bouldin Index': davies_bouldin_scores
})

# DataFrame for Calinski-Harabasz Index
calinski_data = pd.DataFrame({
    'Number of Clusters (k)': list(k_range),
    'Calinski-Harabasz Index': calinski_harabasz_scores
})


# ============================================================================
# 4. VISUALIZE OPTIMIZATION METRICS
# ============================================================================
print("\nSTEP 4: Visualizing Optimization Metrics")

# Plot 1: Elbow Method (WSS)
ax = sns.lineplot(data=elbow_data, x='Number of Clusters (k)', y='WSS',
                  marker='o', markersize=10, linewidth=2.5, color='#1f77b4')
# Customize the plot
ax.set_xlabel('Number of Clusters (k)', fontsize=13, fontweight='semibold')
ax.set_ylabel('Within-Cluster Sum of Squares (WSS)', fontsize=13, fontweight='semibold')
ax.set_title('Elbow Method For Optimal k', fontsize=15, fontweight='bold', pad=20)

# Set x-axis ticks
ax.set_xticks(k_range)

# Enhance grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.savefig('kmeans-wss.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Silhouette Score (higher is better)
ax = sns.lineplot(data=silhouette_data, x='Number of Clusters (k)', y='Silhouette Score',
                  marker='o', markersize=10, linewidth=2.5, color='#2ca02c')

# Customize the plot
ax.set_xlabel('Number of Clusters (k)', fontsize=13, fontweight='semibold')
ax.set_ylabel('Silhouette Score', fontsize=13, fontweight='semibold')
ax.set_title('Silhouette Score vs k (Higher is Better)', fontsize=15, fontweight='bold', pad=20)

# Set x-axis ticks
ax.set_xticks(k_range)

# Enhance grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig('kmeans-silh.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Davies-Bouldin Index (lower is better)
ax = sns.lineplot(data=dbi_data, x='Number of Clusters (k)', y='Davies-Bouldin Index',
                  marker='o', markersize=10, linewidth=2.5, color='#d62728')

# Customize the plot
ax.set_xlabel('Number of Clusters (k)', fontsize=13, fontweight='semibold')
ax.set_ylabel('Davies-Bouldin Index', fontsize=13, fontweight='semibold')
ax.set_title('Davies-Bouldin Index vs k (Lower is Better)', fontsize=15, fontweight='bold', pad=20)

# Set x-axis ticks
ax.set_xticks(k_range)

# Enhance grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.savefig('kmeans-dbi.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 4: Calinski-Harabasz Index (higher is better)
ax = sns.lineplot(data=calinski_data, x='Number of Clusters (k)', y='Calinski-Harabasz Index',
                  marker='o', markersize=10, linewidth=2.5, color='#9467bd')

# Customize the plot
ax.set_xlabel('Number of Clusters (k)', fontsize=13, fontweight='semibold')
ax.set_ylabel('Calinski-Harabasz Index', fontsize=13, fontweight='semibold')
ax.set_title('Calinski-Harabasz Index vs k (Higher is Better)', fontsize=15, fontweight='bold', pad=20)

# Set x-axis ticks
ax.set_xticks(k_range)

# Enhance grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.savefig('kmeans-chi.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 5. TRAIN FINAL MODEL WITH OPTIMAL K
# ============================================================================
print("\nSTEP 5: Training Final K-Means Model")

# Based on the seed dataset, we expect 3 varieties
optimal_k = 3
print(f"\nUsing k={optimal_k} clusters, based on method to find optimal K")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_scaled)

print(f"\nCluster distribution:")
unique, counts = np.unique(cluster_labels, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"  Cluster {cluster}: {count} samples ({count / len(cluster_labels) * 100:.1f}%)")


# ============================================================================
# 5a. CLUSTER CENTROID PROFILES
# ============================================================================
print("\nSTEP 5a: Cluster Centroid Analysis")

# Cluster means in scaled space
centroid_df = pd.DataFrame(kmeans_final.cluster_centers_,
                           columns=feature_names,
                           index=[f'Cluster {i}' for i in range(optimal_k)])

print("\nCluster centroids (scaled feature values):")
print(centroid_df.round(3))

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(centroid_df.T, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, linewidths=0.5, ax=ax, annot_kws={'size': 9})
ax.set_title('Cluster Centroids — Mean Scaled Feature Values',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Cluster', fontsize=11)
ax.set_ylabel('Feature', fontsize=11)
plt.tight_layout()
plt.savefig('kmeans-centroids.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 6. INTRINSIC VALIDATION METRICS
# ============================================================================
print("\nSTEP 6: Intrinsic Validation Metrics (Clustering Quality)")

silhouette = silhouette_score(X_scaled, cluster_labels)
davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)

print(f"\nSilhouette Score: {silhouette:.2f}")
print(f"  Range: [-1, 1], Higher is better")
print(
    f"  Interpretation: {silhouette:.2f} indicates {'excellent' if silhouette > 0.7 else 'good' if silhouette > 0.5 else 'moderate' if silhouette > 0.25 else 'poor'} separation")

print(f"\nDavies-Bouldin Index: {davies_bouldin:.2f}")
print(f"  Range: [0, ∞), Lower is better")
print(
    f"  Interpretation: {davies_bouldin:.2f} indicates {'excellent' if davies_bouldin < 0.5 else 'good' if davies_bouldin < 1.0 else 'moderate' if davies_bouldin < 1.5 else 'poor'} cluster separation")

print(f"\nCalinski-Harabasz Index: {calinski_harabasz:.2f}")
print(f"  Range: [0, ∞), Higher is better")
print(f"  Interpretation: Higher values indicate denser and better separated clusters")

print(f"\nWSS value: {kmeans_final.inertia_:.2f}")

# ============================================================================
# 6a. PER-SAMPLE SILHOUETTE PLOT
# ============================================================================
print("\nSTEP 6a: Per-Sample Silhouette Analysis")

silhouette_vals = silhouette_samples(X_scaled, cluster_labels)
y_lower = 10
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

fig, ax = plt.subplots(figsize=(8, 6))
for i in range(optimal_k):
    cluster_sil = np.sort(silhouette_vals[cluster_labels == i])
    size = cluster_sil.shape[0]
    y_upper = y_lower + size
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                     alpha=0.7, color=colors[i], label=f'Cluster {i}')
    ax.text(-0.05, y_lower + 0.5 * size, str(i), fontsize=10, fontweight='bold')
    y_lower = y_upper + 10

avg_score = silhouette_vals.mean()
ax.axvline(x=avg_score, color='red', linestyle='--', linewidth=1.5,
           label=f'Mean = {avg_score:.3f}')
ax.set_xlabel('Silhouette Coefficient', fontsize=11)
ax.set_ylabel('Cluster', fontsize=11)
ax.set_yticks([])
ax.set_title('Per-Sample Silhouette Plot (K=3)', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.set_xlim(-0.35, 1.0)
plt.tight_layout()
plt.savefig('kmeans-silhouette-detail.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 7. EXTRINSIC VALIDATION METRICS (Compare with Ground Truth)
# ============================================================================
print("\nSTEP 7: Extrinsic Validation Metrics (Comparison with True Labels)")

ari = adjusted_rand_score(y, cluster_labels)
homogeneity = homogeneity_score(y, cluster_labels)
completeness = completeness_score(y, cluster_labels)
v_measure = v_measure_score(y, cluster_labels)

print(f"\nAdjusted Rand Index (ARI): {ari:.3f}")
print(f"  Range: [-1, 1], 1 = perfect match, 0 = random labeling")
print(
    f"  Interpretation: {ari:.3f} indicates {'excellent' if ari > 0.9 else 'very good' if ari > 0.7 else 'good' if ari > 0.5 else 'moderate' if ari > 0.3 else 'poor'} agreement with true labels")

print(f"\nHomogeneity Score: {homogeneity:.3f}")
print(f"  Range: [0, 1], Higher is better")
print(f"  Measures if clusters contain only members of a single class")

print(f"\nCompleteness Score: {completeness:.3f}")
print(f"  Range: [0, 1], Higher is better")
print(f"  Measures if all members of a class are in the same cluster")

print(f"\nV-Measure Score: {v_measure:.3f}")
print(f"  Range: [0, 1], Higher is better")
print(f"  Harmonic mean of homogeneity and completeness")

# ============================================================================
# 8. PCA EXPLAINED VARIANCE (SCREE PLOT)
# ============================================================================
print("\nSTEP 8: PCA Explained Variance")

pca_full = PCA(n_components=7)
pca_full.fit(X_scaled)

explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

print("\nVariance explained per component:")
for i, (e, c) in enumerate(zip(explained, cumulative)):
    print(f"  PC{i+1}: {e*100:.1f}%  (cumulative: {c*100:.1f}%)")

fig, ax1 = plt.subplots(figsize=(9, 5))
bars = ax1.bar(range(1, 8), explained * 100, color='steelblue', alpha=0.7, label='Individual')
ax1.set_xlabel('Principal Component', fontsize=11)
ax1.set_ylabel('Explained Variance (%)', fontsize=11, color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')

# Label each bar
for bar, val in zip(bars, explained * 100):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

ax2 = ax1.twinx()
ax2.plot(range(1, 8), cumulative * 100, 'o-', color='coral',
         linewidth=2, label='Cumulative')
ax2.axhline(y=85, color='grey', linestyle='--', linewidth=1, alpha=0.7, label='85%')
ax2.axhline(y=95, color='grey', linestyle=':', linewidth=1, alpha=0.7, label='95%')
ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=11, color='coral')
ax2.tick_params(axis='y', labelcolor='coral')
ax2.set_ylim(0, 108)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)
ax1.set_title('PCA – Explained Variance by Component', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('kmeans-pca-variance.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 8a. VISUALIZE CLUSTERING RESULTS
# ============================================================================

print("\nSTEP 8a: Visualizing Clustering Results")

# Apply PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"\nExplained variance by first 2 PCs: {sum(pca.explained_variance_ratio_) * 100:.2f}%")

# Plot 1: K-Means Clustering Results
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels,
           cmap='viridis', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
centers_pca = pca.transform(kmeans_final.cluster_centers_)
ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X',
           s=300, edgecolors='black', linewidth=2, label='Centroids')
ax.set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0] * 100:.1f}%)', fontsize=12)
ax.set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1] * 100:.1f}%)', fontsize=12)
ax.set_title('K-Means Clustering Results (PCA Projection)', fontsize=14, fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig('kmeans-pca.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: True Labels (Ground Truth)
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y,
           cmap='plasma', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
ax.set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0] * 100:.1f}%)', fontsize=12)
ax.set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1] * 100:.1f}%)', fontsize=12)
ax.set_title('True Labels (Ground Truth)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('kmeans-truth.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 8b. FEATURE PAIRPLOT BY CLUSTER
# ============================================================================
print("\nSTEP 8b: Feature Pair Relationships by Cluster")

# Build a scaled DataFrame for plotting
scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
scaled_df['Cluster'] = cluster_labels.astype(str)

palette = {'0': '#1f77b4', '1': '#ff7f0e', '2': '#2ca02c'}

g = sns.pairplot(scaled_df, hue='Cluster', palette=palette,
                 diag_kind='kde', plot_kws={'alpha': 0.4, 's': 20})
g.figure.suptitle('Feature Pair Relationships by Cluster (Scaled)',
                  y=1.02, fontsize=13, fontweight='bold')
g.figure.savefig('kmeans-pairplot.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 9. CONFUSION-STYLE COMPARISON
# ============================================================================

print("\nSTEP 9: Cluster-to-Class Mapping")


# Create a contingency table
contingency_table = pd.crosstab(y, cluster_labels,
                                rownames=['True Class'],
                                colnames=['Predicted Cluster'])
print("\nContingency Table (True Classes vs Predicted Clusters):")
print(contingency_table)

# Visualize contingency table
plt.figure(figsize=(8, 6))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues',
            cbar_kws={'label': 'Count'}, linewidths=0.5)
plt.title('Contingency Table: True Classes vs Predicted Clusters',
          fontsize=14, fontweight='bold')
plt.ylabel('True Class', fontsize=12)
plt.xlabel('Predicted Cluster', fontsize=12)
plt.tight_layout()
plt.savefig('kmeans-contingency.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 10. SUMMARY REPORT
# ============================================================================

print("\nFINAL SUMMARY REPORT")


summary = f"""
Dataset Information:
- Total samples: {len(X)}
- Number of features: {X.shape[1]}
- Number of true classes: {len(np.unique(y))}
- Optimal k selected: {optimal_k}

Intrinsic Metrics (Clustering Quality):
- Silhouette Score: {silhouette:.4f} (Higher is better, range: [-1, 1])
- Davies-Bouldin Index: {davies_bouldin:.4f} (Lower is better, range: [0, ∞))
- Calinski-Harabasz Index: {calinski_harabasz:.4f} (Higher is better, range: [0, ∞))

Extrinsic Metrics (Agreement with Ground Truth):
- Adjusted Rand Index: {ari:.4f} (Range: [-1, 1], 1 = perfect)
- Homogeneity Score: {homogeneity:.4f} (Range: [0, 1], higher is better)
- Completeness Score: {completeness:.4f} (Range: [0, 1], higher is better)
- V-Measure Score: {v_measure:.4f} (Range: [0, 1], higher is better)

Cluster Distribution:
{pd.Series(cluster_labels).value_counts().sort_index().to_string()}

Conclusion:
The K-Means clustering {'successfully' if ari > 0.7 else 'reasonably well' if ari > 0.5 else 'partially'} identified the underlying structure
in the seed dataset with an ARI of {ari:.4f} and Silhouette Score of {silhouette:.4f}.
"""

print(summary)

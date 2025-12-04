import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score,
                             homogeneity_score, completeness_score, v_measure_score)
from sklearn.decomposition import PCA
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
plt.rcParams['figure.figsize'] = (12, 8)

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
# 2. DATA PREPROCESSING
# ============================================================================
print("STEP 2: Data Preprocessing (Standardization)")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nScaled data - Mean: {X_scaled.mean(axis=0)}")
print(f"Scaled data - Std: {X_scaled.std(axis=0)}")

# # ============================================================================
# # 3. ELBOW METHOD - FINDING OPTIMAL K
# # ============================================================================
# print("STEP 3: Elbow Method - Finding Optimal K")
#
# # Calculate WSS (Within-Cluster Sum of Squares) for different k values
# k_range = range(2, 11)
# wss = []
# silhouette_scores = []
# davies_bouldin_scores = []
# calinski_harabasz_scores = []
#
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     kmeans.fit(X_scaled)
#     wss.append(kmeans.inertia_)
#
#     # Calculate intrinsic metrics
#     labels = kmeans.labels_
#     silhouette_scores.append(silhouette_score(X_scaled, labels))
#     davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels))
#     calinski_harabasz_scores.append(calinski_harabasz_score(X_scaled, labels))
#
# print("\nWSS values for different k:")
# for k, wss_val in zip(k_range, wss):
#     print(f"k={k}: WSS={wss_val:.2f}")
#
# # ============================================================================
# # 4. VISUALIZE OPTIMIZATION METRICS
# # ============================================================================
# print("STEP 4: Visualizing Optimization Metrics")
#
# fig, axes = plt.subplots(2, 2, figsize=(15, 12))
#
# # Plot 1: Elbow Method (WSS)
# axes[0, 0].plot(k_range, wss, 'bo-', linewidth=2, markersize=8)
# axes[0, 0].set_xlabel('Number of Clusters (k)', fontsize=12)
# axes[0, 0].set_ylabel('Within-Cluster Sum of Squares (WSS)', fontsize=12)
# axes[0, 0].set_title('Elbow Method For Optimal k', fontsize=14, fontweight='bold')
# axes[0, 0].grid(True, alpha=0.3)
# axes[0, 0].set_xticks(k_range)
#
# # Plot 2: Silhouette Score (higher is better)
# axes[0, 1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
# axes[0, 1].set_xlabel('Number of Clusters (k)', fontsize=12)
# axes[0, 1].set_ylabel('Silhouette Score', fontsize=12)
# axes[0, 1].set_title('Silhouette Score vs k (Higher is Better)', fontsize=14, fontweight='bold')
# axes[0, 1].grid(True, alpha=0.3)
# axes[0, 1].set_xticks(k_range)
#
# # Plot 3: Davies-Bouldin Index (lower is better)
# axes[1, 0].plot(k_range, davies_bouldin_scores, 'ro-', linewidth=2, markersize=8)
# axes[1, 0].set_xlabel('Number of Clusters (k)', fontsize=12)
# axes[1, 0].set_ylabel('Davies-Bouldin Index', fontsize=12)
# axes[1, 0].set_title('Davies-Bouldin Index vs k (Lower is Better)', fontsize=14, fontweight='bold')
# axes[1, 0].grid(True, alpha=0.3)
# axes[1, 0].set_xticks(k_range)
#
# # Plot 4: Calinski-Harabasz Index (higher is better)
# axes[1, 1].plot(k_range, calinski_harabasz_scores, 'mo-', linewidth=2, markersize=8)
# axes[1, 1].set_xlabel('Number of Clusters (k)', fontsize=12)
# axes[1, 1].set_ylabel('Calinski-Harabasz Index', fontsize=12)
# axes[1, 1].set_title('Calinski-Harabasz Index vs k (Higher is Better)', fontsize=14, fontweight='bold')
# axes[1, 1].grid(True, alpha=0.3)
# axes[1, 1].set_xticks(k_range)
#
# plt.tight_layout()
# plt.savefig('optimization_metrics.png', dpi=300, bbox_inches='tight')
# plt.show()
#
# # ============================================================================
# # 5. TRAIN FINAL MODEL WITH OPTIMAL K
# # ============================================================================
# print("STEP 5: Training Final K-Means Model")
#
# # Based on the seed dataset, we expect 3 varieties
# optimal_k = 3
# print(f"\nUsing k={optimal_k} clusters (based on known varieties)")
#
# kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
# cluster_labels = kmeans_final.fit_predict(X_scaled)
#
# print(f"\nCluster centers shape: {kmeans_final.cluster_centers_.shape}")
# print(f"Cluster distribution:")
# unique, counts = np.unique(cluster_labels, return_counts=True)
# for cluster, count in zip(unique, counts):
#     print(f"  Cluster {cluster}: {count} samples ({count / len(cluster_labels) * 100:.1f}%)")
#
# # ============================================================================
# # 6. INTRINSIC VALIDATION METRICS
# # ============================================================================
# print("STEP 6: Intrinsic Validation Metrics (Clustering Quality)")
#
# silhouette = silhouette_score(X_scaled, cluster_labels)
# davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
# calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
#
# print(f"\nSilhouette Score: {silhouette:.4f}")
# print(f"  → Range: [-1, 1], Higher is better")
# print(
#     f"  → Interpretation: {silhouette:.4f} indicates {'excellent' if silhouette > 0.7 else 'good' if silhouette > 0.5 else 'moderate' if silhouette > 0.25 else 'poor'} separation")
#
# print(f"\nDavies-Bouldin Index: {davies_bouldin:.4f}")
# print(f"  → Range: [0, ∞), Lower is better")
# print(
#     f"  → Interpretation: {davies_bouldin:.4f} indicates {'excellent' if davies_bouldin < 0.5 else 'good' if davies_bouldin < 1.0 else 'moderate' if davies_bouldin < 1.5 else 'poor'} cluster separation")
#
# print(f"\nCalinski-Harabasz Index: {calinski_harabasz:.4f}")
# print(f"  → Range: [0, ∞), Higher is better")
# print(f"  → Interpretation: Higher values indicate denser and better separated clusters")
#
# # ============================================================================
# # 7. EXTRINSIC VALIDATION METRICS (Compare with Ground Truth)
# # ============================================================================
# print("STEP 7: Extrinsic Validation Metrics (Comparison with True Labels)")
#
# ari = adjusted_rand_score(y, cluster_labels)
# homogeneity = homogeneity_score(y, cluster_labels)
# completeness = completeness_score(y, cluster_labels)
# v_measure = v_measure_score(y, cluster_labels)
#
# print(f"\nAdjusted Rand Index (ARI): {ari:.4f}")
# print(f"  → Range: [-1, 1], 1 = perfect match, 0 = random labeling")
# print(
#     f"  → Interpretation: {ari:.4f} indicates {'excellent' if ari > 0.9 else 'very good' if ari > 0.7 else 'good' if ari > 0.5 else 'moderate' if ari > 0.3 else 'poor'} agreement with true labels")
#
# print(f"\nHomogeneity Score: {homogeneity:.4f}")
# print(f"  → Range: [0, 1], Higher is better")
# print(f"  → Measures if clusters contain only members of a single class")
#
# print(f"\nCompleteness Score: {completeness:.4f}")
# print(f"  → Range: [0, 1], Higher is better")
# print(f"  → Measures if all members of a class are in the same cluster")
#
# print(f"\nV-Measure Score: {v_measure:.4f}")
# print(f"  → Range: [0, 1], Higher is better")
# print(f"  → Harmonic mean of homogeneity and completeness")
#
# # ============================================================================
# # 8. VISUALIZE CLUSTERING RESULTS
# # ============================================================================

# print("STEP 8: Visualizing Clustering Results")
#
# # Apply PCA for 2D visualization
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)
#
# print(f"\nExplained variance by first 2 PCs: {sum(pca.explained_variance_ratio_) * 100:.2f}%")
#
# # Create visualization
# fig, axes = plt.subplots(1, 2, figsize=(16, 6))
#
# # Plot 1: Clustering Results
# scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels,
#                            cmap='viridis', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
# # Plot cluster centers
# centers_pca = pca.transform(kmeans_final.cluster_centers_)
# axes[0].scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X',
#                 s=300, edgecolors='black', linewidth=2, label='Centroids')
# axes[0].set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0] * 100:.1f}%)', fontsize=12)
# axes[0].set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1] * 100:.1f}%)', fontsize=12)
# axes[0].set_title('K-Means Clustering Results (PCA Projection)', fontsize=14, fontweight='bold')
# axes[0].legend()
# plt.colorbar(scatter1, ax=axes[0], label='Cluster')
#
# # Plot 2: True Labels
# scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y,
#                            cmap='plasma', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
# axes[1].set_xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0] * 100:.1f}%)', fontsize=12)
# axes[1].set_ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1] * 100:.1f}%)', fontsize=12)
# axes[1].set_title('True Labels (Ground Truth)', fontsize=14, fontweight='bold')
# plt.colorbar(scatter2, ax=axes[1], label='True Class')
#
# plt.tight_layout()
# plt.savefig('clustering_visualization.png', dpi=300, bbox_inches='tight')
# plt.show()
#
# # ============================================================================
# # 9. CONFUSION-STYLE COMPARISON
# # ============================================================================
# 
# print("STEP 9: Cluster-to-Class Mapping")
#
#
# # Create a contingency table
# contingency_table = pd.crosstab(y, cluster_labels,
#                                 rownames=['True Class'],
#                                 colnames=['Predicted Cluster'])
# print("\nContingency Table (True Classes vs Predicted Clusters):")
# print(contingency_table)
#
# # Visualize contingency table
# plt.figure(figsize=(8, 6))
# sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues',
#             cbar_kws={'label': 'Count'}, linewidths=0.5)
# plt.title('Contingency Table: True Classes vs Predicted Clusters',
#           fontsize=14, fontweight='bold')
# plt.ylabel('True Class', fontsize=12)
# plt.xlabel('Predicted Cluster', fontsize=12)
# plt.tight_layout()
# plt.savefig('contingency_table.png', dpi=300, bbox_inches='tight')
# plt.show()
#
# # ============================================================================
# # 10. SUMMARY REPORT
# # ============================================================================
# 
# print("FINAL SUMMARY REPORT")
#
#
# summary = f"""
# Dataset Information:
# - Total samples: {len(X)}
# - Number of features: {X.shape[1]}
# - Number of true classes: {len(np.unique(y))}
# - Optimal k selected: {optimal_k}
#
# Intrinsic Metrics (Clustering Quality):
# - Silhouette Score: {silhouette:.4f} (Higher is better, range: [-1, 1])
# - Davies-Bouldin Index: {davies_bouldin:.4f} (Lower is better, range: [0, ∞))
# - Calinski-Harabasz Index: {calinski_harabasz:.4f} (Higher is better, range: [0, ∞))
#
# Extrinsic Metrics (Agreement with Ground Truth):
# - Adjusted Rand Index: {ari:.4f} (Range: [-1, 1], 1 = perfect)
# - Homogeneity Score: {homogeneity:.4f} (Range: [0, 1], higher is better)
# - Completeness Score: {completeness:.4f} (Range: [0, 1], higher is better)
# - V-Measure Score: {v_measure:.4f} (Range: [0, 1], higher is better)
#
# Cluster Distribution:
# {pd.Series(cluster_labels).value_counts().sort_index().to_string()}
#
# Conclusion:
# The K-Means clustering {'successfully' if ari > 0.7 else 'reasonably well' if ari > 0.5 else 'partially'} identified the underlying structure
# in the seed dataset with an ARI of {ari:.4f} and Silhouette Score of {silhouette:.4f}.
# """
#
# print(summary)
#
# 
# print("PIPELINE COMPLETED SUCCESSFULLY")
#
# print("\nGenerated files:")
# print("  - optimization_metrics.png")
# print("  - clustering_visualization.png")
# print("  - contingency_table.png")
---

layout: default

title: Wheat Seeds (K-Means Clustering)

permalink: /k-means-clustering/

---

## Goals and objectives:

The objective of this project is to apply **K-Means Clustering** to a dataset of geometric measurements taken from wheat seeds, with the goal of discovering whether the data naturally separates into meaningful groups — without using the known seed variety labels during the analysis.

The dataset contains 210 seeds drawn from three varieties: **Kama**, **Rosa**, and **Canadian**. For each seed, seven geometric properties were measured using a soft X-ray imaging technique: area, perimeter, compactness, kernel length, kernel width, asymmetry coefficient, and groove length. These seven features form the basis of the clustering.
The analysis is structured around three questions:

* **Can K-Means recover the true variety structure from geometry alone?** The model is trained entirely without labels, making this a test of whether the geometric measurements carry enough discriminatory signal to distinguish the varieties.
* **What is the optimal number of clusters?** Rather than assuming three clusters because we know there are three varieties, the optimal K is determined objectively using four independent methods: the Elbow Method (WSS), Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.
* **How well do the recovered clusters align with the true varieties?** Because the true labels are available (but withheld from the model), both intrinsic metrics (which measure clustering quality independently) and extrinsic metrics (which compare against ground truth) are used to assess performance.

The analysis also includes an examination of the feature space: checking for skewness, multicollinearity between features, and using Principal Component Analysis (PCA) to reduce the seven dimensions to two for visualisation. The PCA scree plot quantifies how much variance is captured at each component, directly supporting the interpretation of the cluster scatter plots.

The project concludes by assessing where and why the clustering falls short of perfect agreement with the true labels — specifically exploring the natural overlap between varieties in the feature space — and identifies clear next steps, including a transition to supervised learning methods which leverage the available labels directly.


## Application:  

K-Means Clustering is an unsupervised machine learning algorithm that partitions observations into **K distinct, non-overlapping groups** based on similarity. The algorithm works iteratively: it assigns each observation to its nearest cluster centroid, then recalculates the centroid positions based on the new assignments, repeating until the assignments stabilise. The result is a set of clusters where observations within a group are as similar as possible to each other, and as different as possible from observations in other groups.

A key practical decision in any K-Means analysis is selecting the optimal value of K. This project determines K objectively using four complementary methods — the **Elbow Method** (minimising Within-Cluster Sum of Squares), **Silhouette Score**, **Davies-Bouldin Index**, and **Calinski-Harabasz Index** — with convergence across all four providing confidence in the chosen value.

Because K-Means operates purely on distance in feature space, it requires **scaled data** to prevent features with larger absolute ranges dominating the clustering. It is also worth noting that K-Means assumes clusters are roughly spherical and of similar density; where these assumptions do not hold, alternative methods such as DBSCAN or Gaussian Mixture Models may be more appropriate.

K-Means is one of the most widely deployed clustering techniques across industry, with applications including:

🏦 **Financial Services** — customer segmentation for targeted products, transaction clustering for fraud detection, and risk tiering of financial instruments.  
🛍️ **Retail** — grouping customers by purchasing behaviour to inform marketing strategy, and clustering product lines to optimise inventory management.  
🏭 **Manufacturing** — anomaly detection in sensor data to identify equipment at risk, and grouping operational states to support process optimisation and preventative maintenance.  
💻 **Technology** — organising content for recommendation systems, clustering search queries to improve relevance, and load balancing by distributing workloads across server clusters based on utilisation.  
🌾 **Agriculture** — the focus of this project — grouping crops by measurable physical properties to support variety classification, quality control, and seed selection, without requiring manual expert labelling of every sample.  

## Methodology:  

The analysis was developed in Python using Scikit-learn, Pandas, and NumPy, with visualisations produced using Matplotlib and Seaborn. The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/dongeorge/seed-from-uci), originally compiled from the UCI Machine Learning Repository.

The dataset contains **210 observations** across **seven continuous features**: area (A), perimeter (P), compactness (C), length of kernel (LK), width of kernel (WK), asymmetry coefficient (A_Coef) and length of kernel groove (LKG). A target label identifying the wheat variety (Kama, Rosa, or Canadian) is present in the data but was withheld from the model throughout — used only at the evaluation stage to assess clustering quality against ground truth.

The workflow proceeded through the following stages:

**Data Validation and Exploratory Analysis** The dataset was checked for missing values, data types, and class balance across the three varieties. Feature distributions were examined and skewness quantified for each variable, since K-Means relies on Euclidean distance and can be sensitive to heavily skewed features. A correlation analysis was also conducted to identify highly correlated features, as multicollinearity effectively over-weights certain dimensions in the distance calculation.

**Feature Scaling** All features were standardised using StandardScaler, transforming each to zero mean and unit variance. This is a prerequisite for K-Means, ensuring no single feature dominates the clustering due to differences in scale or units.

**Determining the Optimal Number of Clusters** Rather than assuming K=3 from domain knowledge, the optimal K was determined objectively by running K-Means across K = 2 to 10 and evaluating four independent metrics at each value:

* **Within-Cluster Sum of Squares (WSS)** — used in the Elbow Method to identify the point of diminishing returns
* **Silhouette Score** — measures how similar each observation is to its own cluster versus neighbouring clusters
* **Davies-Bouldin Index** — assesses the ratio of within-cluster scatter to between-cluster separation
* **Calinski-Harabasz Index** — evaluates cluster density and separation simultaneously

Convergence across all four metrics on K=3 provided objective confirmation of the optimal cluster count.

**Model Training** The final K-Means model was trained with K=3, random_state=42 for reproducibility, and n_init=10 — meaning the algorithm was run ten times with different centroid initialisations, with the best result selected. Cluster centroids were extracted and analysed to characterise the defining features of each cluster.

**Validation** Clustering quality was assessed using both intrinsic and extrinsic metrics. Intrinsic metrics (Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index) evaluate cluster structure independently of the true labels. Extrinsic metrics (Adjusted Rand Index, Homogeneity, Completeness, and V-Measure) compare the clustering result directly against the withheld variety labels to quantify agreement with ground truth.

**Visualisation** Principal Component Analysis (PCA) was applied to the scaled data to enable visualisation of the seven-dimensional feature space. A full seven-component PCA was first used to produce a scree plot quantifying the variance captured at each component. A two-component PCA was then used to project the data into two dimensions, generating scatter plots of the K-Means cluster assignments alongside the true variety labels for direct visual comparison. A per-sample silhouette plot and a pairplot of cluster assignments across all feature pairs were also produced to support deeper interpretation of the clustering structure.

## Results and conclusions:

Initially the data was scaled to support optimal clustering, and prevent single factors dominating the clustering process.  The clustering transformed the values, so that each IV has a mean of 0 and a standard deviation equal to 1.

The K-Means Cluster model was run for each value of K in the range [2, 11].

For each value of K, the WSS (Within-Cluster Sum of Squares) was calculated, which were used to determine the optimal value of K, i.e. the optimal number of clusters.  

For each value of K intrinsic measures were also measured and recorded; Silhouette Scores, Davies Bouldin Scores, and Calinski Harabasz Scores.  

![WSS - Elbow](kmeans-wss.png)

![Silhouette](kmeans-silh.png)

![Davies_Bouldin](kmeans-dbi.png)

![Calinski_Harabasz](kmeans-chi.png)

The Elbow Method, which uses the WSS scores for each K value, along with the plots of the other intrinsic measures, was used to determine the optimal K value of 3, which was then used to define the final K-Means Model, and apply the clustering to the observations.

### Intrinsic Validation Metrics

Where K=3 the K-Means Model generated the following metrics:

WSS = 430.66

Silhouette Score: 0.40  
  Range: [-1, 1], Higher is better  
  Interpretation: 0.40 indicates moderate separation  

Davies-Bouldin Index: 0.9279  
  Range: [0, ∞), Lower is better  
  Interpretation: 0.93 indicates good cluster separation  

Calinski-Harabasz Index: 249.78  
  Range: [0, ∞), Higher is better  
  Interpretation: Higher values indicate denser and better separated clusters

### Extrinsic Validation Metrics

This project is an unsupervised learning project as the model was generated unsing unlabelled data, however the data used did have labels (they just weren't presented to the K-Means Clustering model).

As such extrinsic validation metrics can be determined, by comparing the clustering result to the grounded truth.

First it should be noted that the intrinsic validation and elbow method determine the optimal clusters to be 3, which is correct, as there are three varieties of wheat seeds.

Adjusted Rand Index (ARI): 0.773  
  Range: [-1, 1], 1 = perfect match, 0 = random labeling  
  Interpretation: 0.7733 indicates very good agreement with true labels  

Homogeneity Score: 0.728  
  Range: [0, 1], Higher is better - Measures if clusters contain only members of a single class  
  This score indicates that approximately 73% of the time, each cluster contains only samples from a single true class

Completeness Score: 0.728  
  Range: [0, 1], Higher is better - Measures if all members of a class are in the same cluster  
  This score shows that about 73% of samples belonging to the same true class are assigned to the same cluster  

V-Measure Score: 0.728  
  Range: [0, 1], Higher is better - Harmonic mean of homogeneity and completeness  
  Indicates a good performance of the clustering  

### Visualising Clustering Results  

As the independent variables represent 7 dimensions, it is not possible to visualise the clusters in relation to these 7 dimensions.

A technique commonly used for visualising clusters with 4+ dimensions is to utilise Princple Component Analysis (PCA), which won't be described fully within this project, but is covered in a separate project within this portfolio.

In summary PCA (Principal Component Analysis) is a dimensionality reduction technique that:
- Transforms the original 7 features into new uncorrelated variables (components)
- Each principal component is a LINEAR COMBINATION of the original features
- Components are ordered by the amount of variance they explain
- PC1 captures the most variance, PC2 the second most, etc.

Using PCA, the first 2 principle components were identified, which were used to produce a scatter plot of the K-Means generated clusters, as well as a scatter plot of the true groups (or clusters):

![Pca_generated clusters](kmeans-pca.png)

![Pca_actual_clusters](kmeans-truth.png)

Finally the predicted clusters were compared to the actuals, noting that as this was an unsupervised learning method, the model was unaware of the true labels and hence the correct predictions are in the bottom-left to top-right diagonal.

![contingency](kmeans-contingency.png)

### Conclusions:

The fact that all three scores extrinsic metrics (Homogeneity, Completeness and V-Measure) are nearly identical (0.7277, -0.7280) indicates balanced clustering - neither homogeneity nor completeness is significantly better or worse.

These scores collectively suggest that the K-Means algorithm achieved moderately strong alignment with the true seed varieties. The clustering is not perfect (which would be 1.0), but it successfully captures much of the underlying structure in the data. The ~73% agreement indicates that the features used are reasonably predictive of seed variety, howevever, there may be some natural overlap between varieties in the feature space.  

It also highlights that small portion of seeds (~27%) are either misclassified or represent boundary cases that are difficult to distinguish

This level of performance is quite respectable for unsupervised learning, especially considering K-Means had no knowledge of the true labels during training.

Overall the clustering is considered successful, given the evidence that there is natural overlap of observations in the feature space of the true varieties, i.e. there is not a clear separation of true clusters.

Known limitations of this model:
* K-Means assumes spherical clusters (may not match data geometry)
* Unsupervised approach doesn't leverage available labels for training

## Next steps:  

With any analysis it is important to assess how the model and data collection can be improved to better support the business goals.

Recommended next steps include:

#### Transition to Supervised Learning
Having labeled data enables supervised methods will significantly outperform unsupervised clustering, potentially providing higher accuracy and better handling of boundary cases.  This requires creating a dataset with labelled data.
Recommended Algorithms:
* Random Forest Classifier: Handles non-linear relationships, provides feature importance
* Support Vector Machine (SVM): Excellent for finding complex decision boundaries
* Gradient Boosting (XGBoost/LightGBM): State-of-the-art performance for tabular data
* Neural Networks: If pattern complexity requires it, but  likely overkill for 7 features

#### Increasing Sample Size
Collect more samples while ensuring a balanced sampling of varieties.  Look to include boundary examples, and take samples from multiple conditions

#### Feature Engineering
* Feature Analysis
  * Assess PCA loadings
  * Which features have highest loadings on PC1/PC2?
  * Are any features redundant (highly correlated)?
  * Do any features have low variance or discriminatory power?
* Create New Features - Consider Ratios and Interactions of dimensions, for example:
  * Aspect Ratio: Length / Width
  * Shape Index: 4π×Area / Perimeter²
  * Volume Proxy: Area × Kernel Width
* Feature Selection - Identify if redundant features exist
  * Test model performance with reduced feature sets

#### Try Alternative Clustering Methods
* Hierarchical Clustering
* DBSCAN (Density-Based Clustering)
* Gaussian Mixture Models (GMM)
* Ensemble Clustering


## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/K-Mean_Clustering_v2.py)

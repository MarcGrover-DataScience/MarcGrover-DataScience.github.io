---

layout: default

title: Time-data (Moving Averages)

permalink: /moving-averages/

---

## Goals and objectives:

A financial organisation wishes to understand ...

signal extraction (how well the moving average line tracks the real trend)

A soft X-ray technique was used to construct all seven, real-valued attributes (features), for each seed on which the clustering is based.

The results of the K-Means clustering produced good agreement with the known labels, with approximately 73% of seeds classified correctly, with three distinct clusters emerging.  It was also noted that ~85-90% of variance was explained by 2 Principle Components. 

## Application:  

Moving-averages statistical analysis is a fundamental time series technique used to smooth out short-term fluctuations and noise in sequential data, thereby revealing the underlying long-term trends or cycles. It works by calculating a continuously updated average over a fixed-size "window" of consecutive data points; as a new observation enters the calculation, the oldest observation is dropped, causing the average to "move" forward over time. The resulting series of averages creates a line that is less volatile than the original data, making it easier to visually identify and model the general direction of movement.

Moving averages are versatile statistical tools were their real-world benefits span numerous industries by improving decision-making from trading to inventory control, including the following:

* **Finance** - Moving averages are a cornerstone of technical analysis in financial markets, where they are used to interpret asset price movements and trends, supporting buy/sell trading signals.
  * Long-term moving averages, like the 200-day Simple Moving Average (SMA), often act as dynamic levels where asset prices are expected to find support (on a drop) or resistance (on a rise).
* **Technology** - In technology, moving averages help monitor continuous streams of performance data to quickly spot deviations and long-term changes, for example performance trending and anomaly detection.
* **Retail** - Moving averages are a simple yet powerful tool in retail for predicting future needs and managing costs associated with stock.  Demand forecasting uses Smoothed Moving Average (SMMA) to manage stock levels, and influence ordering requirements.
  * The Moving Average Cost (MAC) method is an accounting technique where the cost of goods sold (COGS) is calculated using the constantly updated average cost of all inventory on hand. This stabilises profit margins against fluctuating raw material or acquisition prices.
* **Manufacturing** - In manufacturing, moving averages are essential for maintaining quality and detecting process drift before defects become widespread.  It is used to monitor qualities within the manufacturign process to detect shifts, and support early defect detection.


## Methodology:  

A workflow in Python was developed using libraries Scikit-learn, Pandas and Numpy, utilising Matplotlib and Seaborn for visualisations.  The data used was obtained from [Kaggle](https://www.kaggle.com/datasets/dongeorge/seed-from-uci).  

SMA (Simple Moving Average) , WMA (Weighted Moving Average), EMA (Exponential Moving Average)


It should be noted that when using EMA, there are values from the first time point, whereas for SMA and WMA the first values appear only once a full window of data is observed.  

EMA Uses a recursive formula: EMA_today = α × Price_today + (1-α) × EMA_yesterday , where the first EMA value is typically initialized as the first price itself.  The smoothing factor α = 2/(span+1), so for a 20-day window: α = 2/21 ≈ 0.095



## Results and conclusions:

The Trade-off: Smoothness vs. Lag
When you evaluate a moving average, you are generally trying to find the optimal balance between two competing properties, which can also be quantified:
1. Smoothness (Noise Reduction)
A smoother line has less period-to-period change. Quantification: You can measure the volatility or variance of the moving average line itself. A lower standard deviation of the values in the $\hat{y}_t$ series indicates a smoother line.
2. Lag (Responsiveness)
The smoothed line naturally lags behind the true underlying trend because it incorporates old data. Quantification: This is often measured in time periods as the average difference between the time a significant trend change occurs in the original data and the time the moving average line changes its slope in response. In practice, technical analysts often visually compare a fast EMA (low lag) against a slow SMA (high lag) to demonstrate this trade-off.
The "best" moving average is the one that minimizes the lag while providing enough smoothness to filter out the noise relevant to your analysis (e.g., a 20-day MA is less smooth but less lagged than a 200-day MA).

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
[View the Python Script](/K-Mean_Clustering.py)

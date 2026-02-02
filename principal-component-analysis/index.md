---

layout: default

title: Breast Cancer Dataset (Principal Component Analysis)

permalink: /principal-component-analysis/

---

# This project is in development

Unsupervised learning algorithm  
Scale data  
youtube.com/watch?v=RfbC6fxAxSI  

## Goals and objectives:

The business objective is to predict the cancer status of cells (benign or malignant) based on 30 features of the cells observed via digitised images.  Three previous projects build prediction models using different techniques to achieve high accuracy predictions (decision trees, random forests and gradient boosted trees).  High levels of correlation between some of the features were identified, which lead to recommended analysis into opportunities for dimensionality reduction.  This project is to research the benefits of applying Principal Component Analysis as a technique for deimensionality reduction on the 30 features within the Wisconsin Breast Cancer dataset.

The Wisconsin Breast Cancer dataset is considered a good case for researching and demonstrating PCA because it suffers from extreme redundancy, as highlighted by the high correlation of features. When features are highly correlated, they are essentially telling the same story multiple times.

(Include summary of the findings)

How can the optimal number of Principal Components be determined, as well as what they are?  
What are the outputs other than the components?  
Is there scoring of how much variance is captured / lost?  Is there a measure of quality of the output?

## Application:  

Principal Component Analysis (PCA) is essentially a "space-saving" technique for data. It’s a dimensionality-reduction method that transforms a large set of variables into a smaller one that still contains most of the original information.

An analogy is taking a high-resolution 3D photo of a complex object and finding the perfect 2D angle that captures its shape so well that a user wouldn't even miss the third dimension.

PCA identifies patterns in data based on the correlation between features. It seeks to maximise variance, finding the directions (Principal Components) along which the data is most spread out.  Common steps for applying PCA include:  

* **Standardisation:** Scaling the data so each variable contributes equally.
* **Covariance Matrix Computation:** Identifying how variables vary from the mean with respect to each other.
* **Eigenvector/Eigenvalue Calculation:** Determining the principal components.
* **Feature Vector:** Choosing which components to keep and which to discard.

Benefits of Using PCA include:

* **Dimensionality Reduction:** It simplifies complex datasets, making them easier to explore and visualize (e.g., turning 10 variables into a 2D plot).
* **Noise Reduction:** By discarding components with low variance, you often filter out "noise" and keep the "signal."
* **Improved Algorithm Performance:** Many machine learning algorithms (like regressions or clustering) run faster and more accurately when they aren't bogged down by redundant variables.
* **Feature Correlation:** It eliminates multicollinearity, ensuring that the remaining features are independent of one another.

PCA is a technique that can be applied in multiple scenarios across all business sectors. PCA serves as a "noise filter," allowing professionals to ignore the hundreds of minor variables and focus on the few underlying forces that actually drive results.  Practical real-world examples include:

* **Medical Sector: Genomics & Disease Subtyping** - In modern medicine, a single patient sample can contain data on expression levels for over 20,000 genes. Researchers use PCA to condense these thousands of gene expressions into a few "eigen-genes".  This helps identify distinct patient clusters. For example, in cancer research, PCA can reveal that what looks like one disease is actually three different subtypes that require different treatments, based on how the gene data clusters in multi-dimensional space, where PCA reduces the dimensions simplfying the understanding.

* **Finance Sector: Portfolio Risk Management** - Financial markets contain multiple fluctuating stock prices, interest rates, and commodity values.  Analysts apply PCA to a portfolio of dozens of stocks to find "Common Factors".  Instead of watching many individual stock movements (e.g. 50-100), they watch the first few principal components (e.g. 3-5), which often represent Market Sentiment, Interest Rate Sensitivity, and Industry Trends.  A primary benefit is that PCA simplifies Risk Assessment.  As an example, should the first principal component (e.g. "The General Market") drop, the analyst knows exactly how much of their portfolio is exposed to that specific systematic risk versus individual company risk.

* **Manufacturing Sector: Quality Control & Predictive Maintenance** - High-tech factories use hundreds of sensors to monitor temperature, vibration, pressure, and speed on an assembly line.  PCA aggregates these sensor readings into a a few (or even a single) "Health Score".  This application of PCA supports improved Anomaly Detection. For example, in a 30-sensor system, it’s hard to tell if one sensor is slightly off. However, when PCA combines them, a "drift" in the first principal component can signal that a machine is beginning to fail long before an actual breakdown occurs, allowing for proactive maintenance.

* **Science Sector: Remote Sensing & Climate Study** - Satellite imagery (Hyperspectral imaging) captures data across hundreds of different light wavelengths, many of which are invisible to the human eye. Scientists apply PCA to satellite data of a forest or ocean.  A key benefit of this application of PCA, is Feature Extraction. While a raw image might just look green, PCA can separate the "noise" of sunlight reflection from the "signal" of chlorophyll density or moisture levels. This allows scientists to map deforestation or drought levels with extreme precision using just the top 2 or 3 components.

## Methodology:  

Details of the methodology applied in the project.

The dataset used is the same as used in the Decision Tree, Random Forest and Gradient Boosted Trees projects - the Wisconsin Breast Cancer dataset.  This is available from scikit-learn, including 569 observations, including 30 independent features.

The dataset is also available from Kaggle [here](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)  

* **Dataset validation** to confirm no missing values, and basic descriptive analysis on the features including the correlation between the 30 features. No data pre-processing was undertaken.  
* **Scaling**  the feature data, so that for each feature the mean is zero, with a standard deviation equal to 1.  This is an important step in performing PCA as the technique is based on variance, therefore a variable with a range of 0–1000 will unfairly dominate a variable with a range of 0–1.
* **Identify the top 2 components** for the breast cancer features.  The initial model was to determine the top 2 

## Results and conclusions:

Results from the project related to the business objective.

### Feature Correlation:

Correlation of the 30 features was determine as visualised in the correlation matrix below. This highlights that many of the fields have low-correlation, however there appears to be high-correlation in the features relating to radius, area and perimeter metrics, where the correlation is in the range (0.8, 1.0).  This evidence of high-correlation suggests the implementation of PCA is suitable to this data.

![correlation_matrix](correlation_matrix.png)

### Identifying 2 Principal Components:

The plot below shows the plot of the points for the training data against the top 2 principal components, mapped to the determined diagnosis of malignant or benign.  This plot visualises there is good separation of the dependent variable for the data plotted against the two principal components.

![pca_scatter](pca_scatter.png)

Each of the two principal components are a combination of the features in the original data.  The variance explained by the first 2 principal components is 63.24%.

Visualising PCA is crucial because the components themselves are "abstract", they don't have the simple names (like "mean radius") that original data has. To truly understand them, you need to look at both the variance they capture and the influence of the original variables.

![pca_comp_heatmap](pca_comp_heatmap.png)

### Understanding the Principal Components:

A Biplot is a powerful visualisation in PCA because it bridges the gap between the "abstract" principal components and the "real-world" features.  

A Biplot is a scatter plot with vectors (arrows) overlaid. Each arrow represents an original feature (e.g., area, smoothness).  The direction of the arrow shows which component that feature contributes to most. If an arrow for "Mean Area" points heavily along the X-axis (PC1), then PC1 represents "Size".  Longer arrows indicate that the feature has a stronger influence on that component.

If two arrows are close together, those features are highly correlated.  If arrows are 90° apart, they are uncorrelated.

The Biplot below shows the relation of the top 10 features for the 2 principal components (not all 30 features were included for simplicity).  

![pca_biplot](pca_biplot.png)

From the plot it can be seen that...  arrows for "mean area", "mean perimeter", and "mean radius" all pointing in almost exactly the same direction. This is visual confirmation of the high correlation we discussed earlier—they are effectively providing the same information to the model.

### Optimum number of components:

The Scree Plot (The "How many do I need?" view)
A Scree Plot shows the percentage of total variance explained by each principal component, and is a key analyitical tool to determine the optimum number of components to use.  The initial analysis of principal components looked at the top 2, where the value of 2 was selected arbitrarily.  More thorough analysis of the optimum number of components 

In the Breast Cancer dataset, you'll notice a sharp "elbow" where the variance explained drops off.

What it tells you: It helps you decide the "cut-off" point. If the first 3 components explain 90% of the variance, you can safely ignore the other 27.

The Goal: You want a small number of components to capture a large amount of information.



### Loading Heatmap (contains a lot of what is shown above)

Loading Heatmap (The "Deep Dive" view)
If 30 arrows on a Biplot look too messy, a Heatmap is a cleaner way to see the "recipe" for each component.

How to read it: Each row is a Principal Component, and each column is an original feature.

The Insight: You might find that PC1 is heavily weighted by "Size" features (Area, Perimeter), while PC2 is heavily weighted by "Shape" features (Concavity, Fractal Dimension). This allows you to rename PC1 to "Tumor Bulk" and PC2 to "Tumor Irregularity" in your report.


### Topic to add:  Cumulative Explained Variance

### Conclusions:

Conclusions from the project findings and results.

## Next steps:  

With any analysis it is important to assess how the model and application of the analytical methods can be used and evolved to support the business goals and business decisions and yield tangible benefits.


## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/DecisionTree_BreastCancer.py)

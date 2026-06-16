---

layout: default

title: Breast Cancer Dataset (Principal Component Analysis)

permalink: /principal-component-analysis/

---

## Goals and objectives:

Three previous projects applied supervised classification models (Decision Tree, Random Forest, and Gradient Boosted Trees) to the Wisconsin Breast Cancer dataset, achieving high predictive accuracy across 30 morphological features. High inter-feature correlation was identified in those projects, motivating investigation into dimensionality reduction as a means of simplifying the feature space without significant loss of information.

This project applies Principal Component Analysis (PCA) to the same dataset, with the following objectives:

* **Justify the application of PCA** by validating the presence of high multicollinearity across the 30 features via a correlation matrix.
* **Demonstrate dimensionality reduction** by projecting the data onto two principal components and visualising the resulting class separation between malignant and benign samples.
* **Interpret the principal components** by analysing feature loadings to assign meaningful real-world meaning to the abstract components identified.
* **Determine the optimal number of components** using a Scree Plot and cumulative variance analysis, identifying both the elbow point and a defined variance threshold.

The analysis confirms substantial redundancy across the 30 features: PC1 and PC2 together explain 63.24% of variance, and the Scree Plot identifies an elbow at 3 components (72.64%). For the purposes of demonstration and visualisation, this project focuses on 2 components; in a downstream classification task, 3 or more components would be the recommended starting point.

## Application:  

Principal Component Analysis (PCA) is essentially a "space-saving" technique for data. It’s a dimensionality-reduction method that transforms a large set of variables into a smaller one that still contains most of the original information.  PCA is an unsupervised learning algorithm, as it does not use labels as a target variable, and looks only at the  structure of the input data.  Beyond dimensionality reduction, PCA is widely used for feature extraction, noise filtering, and exploratory data analysis.

An analogy is taking a high-resolution 3D scan of a complex object and finding the 2D projection that preserves the most detail — so that a viewer captures the essential structure without needing the third dimension.

PCA identifies patterns in data based on the correlation between features. It seeks to maximise variance, finding the directions (Principal Components) along which the data is most spread out.  Mathematically, this is achieved by computing the eigenvectors of the feature covariance matrix; each eigenvector defines a principal component direction, and its corresponding eigenvalue quantifies the variance explained in that direction. 

Benefits of Using PCA include:

* **Dimensionality Reduction:** It simplifies complex datasets, making them easier to explore and visualize (e.g., turning 10 variables into a 2D plot).
* **Noise Reduction:** By discarding components with low variance, you often filter out "noise" and keep the "signal."
* **Improved Algorithm Performance:** Many machine learning algorithms (like regressions or clustering) run faster and more accurately when they aren't bogged down by redundant variables.
* **Feature Correlation:** It eliminates multicollinearity, ensuring that the remaining features are independent of one another.

PCA is a technique that can be applied in multiple scenarios across all business sectors. PCA serves as a "noise filter," allowing professionals to ignore the hundreds of minor variables and focus on the few underlying forces that actually drive results.  Practical real-world examples include:

🏥 **Life Sciences: Genomics & Disease Subtyping** - In modern medicine, a single patient sample can contain data on expression levels for over 20,000 genes. Researchers use PCA to condense these thousands of gene expressions into a few "eigen-genes".  This helps identify distinct patient clusters. For example, in cancer research, PCA can reveal that what looks like one disease is actually three different subtypes that require different treatments, based on how the gene data clusters in multi-dimensional space, where PCA reduces the dimensions simplfying the understanding.

🏦 **Finance: Portfolio Risk Management** - Financial markets contain multiple fluctuating stock prices, interest rates, and commodity values.  Analysts apply PCA to a portfolio of dozens of stocks to find "Common Factors".  Instead of monitoring individual stock movements across a large portfolio, analysts watch the first few principal components, which often capture dominant sources of shared variance — such as broad market movements, sector-wide trends, or interest rate sensitivity — though interpreting components in economic terms requires care, as PCA components are defined mathematically, not conceptually.

🏭 **Manufacturing: Quality Control & Predictive Maintenance** - High-tech factories use hundreds of sensors to monitor temperature, vibration, pressure, and speed on an assembly line.  PCA aggregates these sensor readings into a a few (or even a single) "Health Score".  This application of PCA supports improved Anomaly Detection. For example, in a 30-sensor system, it’s hard to tell if one sensor is slightly off. However, when PCA combines them, a "drift" in the first principal component can signal that a machine is beginning to fail long before an actual breakdown occurs, allowing for proactive maintenance.

🏥 **Science: Remote Sensing & Climate Study** - Satellite imagery (Hyperspectral imaging) captures data across hundreds of different light wavelengths, many of which are invisible to the human eye. Scientists apply PCA to satellite data of a forest or ocean.  A key benefit of this application of PCA, is Feature Extraction. While a raw image might just look green, PCA can separate the "noise" of sunlight reflection from the "signal" of chlorophyll density or moisture levels. This allows scientists to map deforestation or drought levels with extreme precision using just the top 2 or 3 components.

## Methodology:  

The dataset used is the Wisconsin Breast Cancer dataset, consistent with the Decision Tree, Random Forest, and Gradient Boosted Trees projects in this portfolio. It is available directly from scikit-learn and comprises 569 observations across 30 numerical features derived from digitised images of fine needle aspirate (FNA) biopsies, with a binary target of Malignant (212) or Benign (357).

The dataset is also available from Kaggle [here](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

The following steps were undertaken:

* **Dataset Validation** — The dataset was confirmed to contain no missing values. Descriptive statistics were reviewed across the 30 features, and the class distribution was confirmed as 212 Malignant (37.3%) and 357 Benign (62.7%).

* **Correlation Analysis** — A correlation matrix was computed across all 30 features and visualised as a lower-triangle heatmap. Pairs with an absolute correlation exceeding 0.90 were identified and printed, providing the analytical justification for applying PCA: where features are highly correlated, they are conveying redundant information that PCA can consolidate.

* **Feature Scaling** — All features were standardised to zero mean and unit variance using `StandardScaler` prior to applying PCA. This step is essential, as PCA is variance-based: without scaling, features with larger numerical ranges would disproportionately dominate the principal components regardless of their true informational value.

* **PCA — 2 Components** — PCA was applied reducing the 30 features to 2 principal components for the purposes of visualisation and demonstration. The resulting projections were plotted as a scatter plot, coloured by diagnosis, to assess class separability in the reduced space.

* **Feature Loadings Analysis** — The contribution of each original feature to PC1 and PC2 was extracted as a loadings matrix and visualised as an annotated heatmap. The top 5 features by absolute loading were identified for each component and printed, providing the basis for interpreting the components in real-world terms.

* **Biplot** — A biplot was produced overlaying the 2-component scatter plot with loading vectors for the first 10 features. This bridges the gap between the abstract components and the original features: arrow direction indicates which component a feature contributes to most, arrow length indicates the strength of that contribution, and arrows pointing in similar directions confirm high inter-feature correlation.

* **Scree Plot and Variance Analysis** — PCA was re-fitted without a component limit to capture the full variance profile across all 30 components. Individual and cumulative explained variance were plotted as a combined Scree Plot. The number of components required to exceed 90% cumulative variance was computed programmatically, and the elbow point was annotated directly on the chart. This analysis determines the optimal number of components for a downstream classification task, independent of the 2-component choice made for visualisation purposes.

## Results:

### Feature Correlation:

A correlation matrix was computed across all 30 features to assess the suitability of PCA prior to its application. The heatmap below reveals substantial multicollinearity, particularly amongst features related to tumour size: radius, perimeter, and area metrics exhibit pairwise correlations in the range (0.95, 1.00) across both mean and worst-case measurements. In total, several feature pairs exceed the threshold of the absolute value of r > 0.9, confirming that the feature space contains significant redundancy that PCA is well-suited to address.

![correlation_matrix](correlation_matrix.png)

### Dimensionality Reduction — 2 Principal Components:

PCA was applied to reduce the 30 standardised features to 2 principal components. The scatter plot below projects all 569 observations onto PC1 and PC2, coloured by diagnosis.

![pca_scatter](pca_scatter.png)

The two classes form distinct, well-separated clusters in the reduced space, with malignant samples concentrated at higher PC1 values and benign samples at lower values. This confirms that the dominant sources of variance in the data align strongly with the diagnostic outcome — a meaningful result given that PCA operates entirely unsupervised, with no access to the class labels during fitting. PC1 and PC2 together account for 63.24% of the total variance in the original 30-feature dataset.

### Feature Loadings and Component Interpretation:

To interpret what PC1 and PC2 represent in real-world terms, the feature loadings were extracted and visualised as an annotated heatmap.

![pca_comp_heatmap](pca_comp_heatmap.png)

The top 5 contributing features by absolute loading for each component are:

**PC1** is dominated by size and shape features — worst area, worst perimeter, worst radius, mean area, and mean perimeter all carry strong positive loadings. This component can reasonably be interpreted as a measure of **Tumour Size and Shape**.

**PC2** is most strongly influenced by texture and irregularity features — mean fractal dimension, worst fractal dimension, and mean smoothness carry the highest loadings, with a different directional profile to PC1. PC2 can be interpreted as a measure of **Tumour Irregularity**.

The biplot below reinforces these interpretations visually, overlaying loading vectors for the first 10 features onto the 2-component scatter plot. Features with arrows pointing in similar directions are highly correlated (confirming the earlier correlation analysis), and the dominant alignment of size-related arrows along the PC1 axis is clearly visible.

![pca_biplot](pca_biplot.png)

### Optimal Number of Components:

The 2-component model was chosen for the purposes of visualisation and demonstration. To determine the optimal number of components for a downstream analytical task, PCA was re-fitted across all 30 components and the explained variance profile examined via a Scree Plot.

![pca_variance_analysis](pca_variance_analysis.png)

The Scree Plot identifies a clear elbow at **3 components**, which together explain **72.64%** of the total variance. Beyond this point, each additional component contributes diminishing marginal variance, suggesting that 3 components represent a practical optimum balancing information retention against dimensionality reduction.

Should a specific variance threshold be required — for example, to retain at least 90% of the information in the original dataset — the cumulative variance curve shows that **7 components** are needed, accounting for 91.01% of total variance. The appropriate choice depends on the requirements of the downstream task.

```
Component Explained Variance
PC1: 44.27% (Cumulative: 44.27%)
PC2: 18.97% (Cumulative: 63.24%)
PC3: 9.39% (Cumulative: 72.64%)
PC4: 6.60% (Cumulative: 79.24%)
PC5: 5.50% (Cumulative: 84.73%)
PC6: 4.02% (Cumulative: 88.76%)
PC7: 2.25% (Cumulative: 91.01%)
PC8: 1.59% (Cumulative: 92.60%)
PC9: 1.39% (Cumulative: 93.99%)
PC10: 1.17% (Cumulative: 95.16%)
```  

## Conclusions:

* **Large Data Redundancy** - The 30 features measured in the Wisconsin dataset are highly redundant.  The first two principal components alone capture approximately 63% of the total variance, and just 7 components reach the 91% mark.  Most of the information needed to describe these tumors is contained in less than a quarter of the variables collected, i.e. the datasets can be reduced from 30 features to 7 components while still retaining 91% of the original information.  The remaining 23 components only contain about 9% of the total variance combined, confirming that most features in this dataset were highly redundant.

* **Structural Separation of Tumor Types** - PCA reveals that the "Malignant" and "Benign" samples are not randomly distributed; they form distinct, separable clusters in reduced space.  The PCA scatter plot shows a clear boundary between the two classes.  This confirms that the underlying physical characteristics of the cells are fundamentally different between the two groups, making this dataset an excellent candidate for machine learning classification.

* **Size and Shape as the Primary Drivers (PC1)** - By examining the loadings, we can conclude that the first principal component (PC1) essentially represents the overall magnitude or "bulk" of the cell nuclei, along with the shape in terms of the concavity.  Features like mean radius, mean perimeter, mean area, and mean concavity all have high, positive loadings on PC1 and their vectors in the Biplot point in the same direction.  The single most important factor distinguishing these samples is how large and concave the cells are. Larger values on the PC1 axis correlate strongly with malignant samples.

* **"Irregularity" as the Secondary Driver (PC2)** - While PC1 focuses on size, PC2 often captures features related to the complexity or irregularity of the cell boundary.  This can be interpretted as the texture of the cell boundary.  Features like smoothness, and fractal dimension often weight heavily on PC2.  After accounting for size, the next most important differentiator is how "deformed" or "rough" the cell edges are. This provides a secondary layer of diagnostic information that size alone might miss.

* **Identification of "Proxy" Features** - The Biplot reveals that many features are virtually identical in the information they provide.  This is evidenced by the arrows for radius, perimeter, and area are almost perfectly overlapping.  In a real-world clinical setting, the data collection process could be simplified by reducing the number of measurements. Instead of meticulously measuring all three, just one could be measured (e.g., mean area) to act as a "proxy" for the others without losing significant diagnostic power.

* **Detection of Transitionary Samples** - Not all points sit deep within their respective clusters; some sit in the "border zone" between Benign and Malignant.  This is represented as the overlap area in the PCA scatter plot.  These points represent "borderline" cases where the cell characteristics are ambiguous.  PCA helps identify these specific samples for further review by a human pathologist, highlighting where the automated model might be less certain.

## Next steps:  

With any analysis it is important to assess how the application of the analytical methods can be used and evolved to support the business goals and business decisions and yield tangible benefits. As such it is important to focus on how the insights gained from PCA can be operationalised. PCA tells you what the data is doing; thereofre the next steps explain how to act on that knowledge to improve clinical outcomes (the ultimate business goal in this scenario).

* **Feature Engineering: The "Shape vs. Size" Ratio** - Since PCA identified that size and shape (PC1) and shape irregularity (PC2) are the primary drivers of variance, a logical next step is to create explicit ratios. For example, a "Circularity Index" ($Area/Perimeter^2$) could consolidate the redundant size metrics into a single, high-impact feature. This allows a machine learning model to focus on the relationship between dimensions rather than just the raw measurements themselves.
* **Implement a Predictive Classifier (Random Forest or SVM)** - Now that PCA has reduced the noise, the next step is to feed the top principal components into a supervised learning algorithm. A Support Vector Machine (SVM) is particularly effective here because PCA has already helped linearise the boundaries between classes. You can compare a model trained on all 30 features versus one trained on the top 7 components to demonstrate that dimensionality reduction often leads to better "generalisation" (the ability to predict new, unseen cases).
* **Cost-Effective Diagnostic Protocols** - In a real-world clinic, time and money are finite. Based on the feature loadings, you can recommend a "Tiered Diagnostic Approach." If the highly influential "Mean Concave Points" and "Worst Area" are within a certain threshold, the tumor is flagged immediately. This minimises the need for pathologists to manually measure the other 28 features for every single patient, streamlining the workflow without sacrificing accuracy.
* **Anomaly Detection for "Edge Cases"** - Using the PCA-transformed space, you can calculate the distance of new patients from the "average" benign or malignant cluster. Patients who fall into the "gray area" or far outside both clusters (outliers) should be flagged for an automatic second opinion or a different type of biopsy. This uses PCA as a safety net to identify cases where the standard metrics might be misleading.
* **Integration of Multi-Modal Data** - The Wisconsin dataset is purely physical (morphological). A major real-world improvement would be to merge this data with Patient History (age, genetics) and Proteomics. You could run a "Multi-Block PCA" to see how physical tumor traits correlate with genetic markers. This provides a holistic view, moving the analysis from "what the tumor looks like" to "how the tumor is behaving."
* **Longitudinal Analysis (Time-Series PCA)** - If data could be collected at multiple time intervals, PCA could be used to track the trajectory of a tumor. By plotting a patient’s "PCA Score" over several months, you could see if the tumor is moving toward the "Malignant" cluster in latent space. This would allow for early intervention based on the direction of change rather than waiting for the tumor to reach a critical size.
* **Hyperparameter Tuning on PCA Components** - If you move forward with a machine learning model, the number of components ($k$) should be treated as a hyperparameter. Instead of just picking 90% variance, you can use Cross-Validation to find the exact number of components that yields the highest F1-score. This ensures the model isn't just mathematically efficient, but clinically optimized to minimize "False Negatives" (missing a malignant tumor).
* **Automated Feature Extraction via Computer Vision** - The current dataset relies on humans or simple software to calculate "mean radius" and "perimeter." A modern next step would be to use Convolutional Neural Networks (CNNs) on raw ultrasound or histology images to extract features automatically. You could then use PCA to compare these "AI-generated" features with traditional "human-defined" features to see which set provides a clearer separation of tumor types.
* **Model Interpretability with SHAP or LIME** - While PCA simplifies the data, it can make models harder to explain to doctors (who might not understand what "PC1" means). Using interpretability tools like SHAP (SHapley Additive exPlanations), you can map the model's decisions back to the original features. This "closes the loop" by showing that the model’s reliance on PC1 is actually a reliance on "Mean Concave Points," making the AI's "black box" transparent and trustworthy for medical professionals.
* **Deployment as a "Decision Support" Tool** - The final real-world step is deploying the PCA-backed model into a web-based dashboard for clinicians. This tool would allow a lab tech to input measurements and see an immediate visual representation of where that patient sits on the PCA Biplot relative to thousands of past cases. This provides a visual, data-driven "second opinion" that helps doctors make more confident diagnoses.

## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/PCA_BreastCancer_v2.py)

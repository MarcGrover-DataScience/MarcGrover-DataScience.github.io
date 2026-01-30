---

layout: default

title: Project (Principle Component Analysis)

permalink: /principle-component-analysis/

---

# This project is in development

## Goals and objectives:

The business objective is to predict the cancer status of cells (benign or malignant) based on 30 features of the cells observed via digitised images.  Three previous projects build prediction models using different techniques to achieve high accuracy predictions (decision trees, random forests and gradient boosted trees).  High levels of correlation between some of the features were identified, which lead to recommended analysis into opportunities for dimensionality reduction.  This project is to research the benefits of applying Principle Component Analysis as a technique for deimensionality reduction on the 30 features within the Wisconsin Breast Cancer dataset.

The Wisconsin Breast Cancer dataset is considered a good case for researching and demonstrating PCA because it suffers from extreme redundancy, as highlighted by the high correlation of features. When features are highly correlated, they are essentially telling the same story multiple times.

(Include summary of the findings)

How can the optimal number of Principle Components be determined, as well as what they are?
What are the outputs other than the components?  Is there scoring of how much variance is captured / lost?  Is there a measure of quality of the output?

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

PCA is a technique that can be applied in multiple scenarios across all business sectors. PCA serves as a "noise filter," allowing professionals to ignore the hundreds of minor variables and focus on the few underlying forces that actually drive results.

1. Medical Sector: Genomics & Disease Subtyping
In modern medicine, a single patient sample can contain data on expression levels for over 20,000 genes.

Application: Researchers use PCA to condense these thousands of gene expressions into a few "eigen-genes."

Benefit: It helps identify distinct patient clusters. For example, in cancer research, PCA can reveal that what looks like one disease is actually three different subtypes that require different treatments, based on how the gene data clusters in 2D or 3D space.

2. Finance Sector: Portfolio Risk Management
Financial markets are a chaotic web of fluctuating stock prices, interest rates, and commodity values.

Application: Analysts apply PCA to a portfolio of dozens of stocks to find "Common Factors." Instead of watching 50 individual stock movements, they watch the first three principal components, which often represent Market Sentiment, Interest Rate Sensitivity, and Industry Trends.

Benefit: It simplifies Risk Assessment. If the first principal component (e.g., "The General Market") drops, the analyst knows exactly how much of their portfolio is exposed to that specific systematic risk versus idiosyncratic (individual company) risk.

3. Manufacturing Sector: Quality Control & Predictive Maintenance
High-tech factories use hundreds of sensors to monitor temperature, vibration, pressure, and speed on an assembly line.

Application: PCA aggregates these sensor readings into a single "Health Score."

Benefit: Anomaly Detection. In a 30-sensor system, it’s hard to tell if one sensor is slightly off. However, when PCA combines them, a "drift" in the first principal component can signal that a machine is beginning to fail long before an actual breakdown occurs, allowing for proactive maintenance.

4. Science Sector: Remote Sensing & Climate Study
Satellite imagery (Hyperspectral imaging) captures data across hundreds of different light wavelengths, many of which are invisible to the human eye.

Application: Scientists apply PCA to satellite data of a forest or ocean.

Benefit: Feature Extraction. While a raw image might just look green, PCA can separate the "noise" of sunlight reflection from the "signal" of chlorophyll density or moisture levels. This allows scientists to map deforestation or drought levels with extreme precision using just the top 2 or 3 components.

Details of how this is applicable to multiple industries to solve business problems, generate insight and provide tangible business benefits. 


## Methodology:  

Details of the methodology applied in the project.

![correlation_matrix](correlation_matrix.png)

Always remember to scale your data before performing PCA. Because PCA is based on variance, a variable with a range of 0–1000 will unfairly dominate a variable with a range of 0–1.

## Results and conclusions:

Results from the project related to the business objective.

### Conclusions:

Conclusions from the project findings and results.

## Next steps:  

With any analysis it is important to assess how the model and application of the analytical methods can be used and evolved to support the business goals and business decisions and yield tangible benefits.


## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/DecisionTree_BreastCancer.py)

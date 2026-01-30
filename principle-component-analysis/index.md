---

layout: default

title: Project (Principle Component Analysis)

permalink: /principle-component-analysis/

---

# This project is in development

## Goals and objectives:

The business objective is to predict the cancer status of cells (benign or malignant) based on 30 features of the cells observed via digitised images. A decision tree model was built to make the predictions, achieving an accuracy of 93.86%, with the most important feature identified as 'worst radius'.

The Wisconsin Breast Cancer dataset is a "textbook" case for PCA because it suffers from extreme redundancy. When features are highly correlated, they are essentially telling the same story multiple times.

## Application:  

Principal Component Analysis (PCA) is essentially a "space-saving" technique for data. It’s a dimensionality-reduction method that transforms a large set of variables into a smaller one that still contains most of the original information.

Think of it like taking a high-resolution 3D photo of a complex object and finding the perfect 2D angle that captures its shape so well you don't even miss the third dimension.

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



Details of how this is applicable to multiple industries to solve business problems, generate insight and provide tangible business benefits. 


## Methodology:  

Details of the methodology applied in the project.

!(correlation_matrix)[correlation_matrix.png]

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

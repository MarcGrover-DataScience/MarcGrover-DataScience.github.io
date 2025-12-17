---

layout: default

title: Project (Decision Trees)

permalink: /decision-trees/

---

#### This project is in development

## Goals and objectives:

The business objective is to predict the cancer status of cells (benign or malignant) based on 30 features of the cells observed via digitised images.  A decision tree model was built to make the predictions, achieving an accuracy of ...

## Application:  

Details of how this is applicable to multiple industries to solve business problems, generate insight and provide tangible business benefits. 


## Methodology:  

Details of the methodology applied in the project.

The dataset used is the Wisconsin Breast Cancer dataset available from scikit-learn, which contains 569 observations, including 30 independent features.

The dataset is also available from Kaggle [here](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

The dataset was validated to confirm that there are no missing values, and basic descriptive analysis was undertaken on the features including the correlation between the 30 features.  No data pre-processing was undertaken.

Decision tree depth analysis was undertaken to determine the optimal depth of the decision tree being created, to generate the most accurate model, and not cause overfitting.


## Results and conclusions:

Simple descriptive analytics determined that 212 observations relate to malignant cancers and 357 relate to benign cancers.

Correlation of the 30 features was undertaken and visualised as a correlation matrix as shown below.  This highlights that many of the fields have low-correlation, however there appears to be high-correlation in the features relating to radius, area and perimeter metrics.  This was not addressed at this stage, but important insight for any future development to improve the predictions.

![correlation](correlation_matrix.png)

Training and testing sets were determined from the 569 observations in the data, where 80% of the data was for training, and the remaining 20% for testing.

Decision tree depth analysis was undertaken on levels in the range (1,13), where for each level the accuracy on the training set, and test set were determined, along with the Cross-Validation score for each number of values, where the number of folds was set to 5.  

The plot below shows the results of the tree depth analysis, which determined that a depth of 3 is optimal, however a depth of 4 also produced similarly accurate results.  This plot also showed that decision trees of 5 or more levels produced less accurate predictions, almost certainly due to over-fitting to the training data.

![depth_analysis](depth_analysis.png)

### Conclusions:

Conclusions from the project findings and results.

## Next steps:  

With any analysis it is important to assess how the model and application of the analytical methods can be used and evolved to support the business goals and business decisions and yield tangible benefits.

* Feature engineering - including removing and / or adding features


## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/DecisionTree_BreastCancer.py)

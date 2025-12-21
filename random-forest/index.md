---

layout: default

title: Breast Cancer Predictions (Random Forest)

permalink: /random-forest/

---

#### This project is in development

## Goals and objectives:

The business objective is to predict the cancer status of cells (benign or malignant) based on 30 features of the cells observed via digitised images. A previous project built a decision tree model achieving an accuracy of 93.86%, and the goal is to research if using a Random Forest predictor can produce more accurate results, and produce more insights into the data supporting the predictions.

This follows on from the Decision Tree project found [here](https://marcgrover-datascience.github.io/decision-trees/)

Add results...

## Application:  

A Random Forest is an ensemble learning method that constructs a multitude of decision trees during training. It is one of the most popular and versatile tools in industry because it follows the "wisdom of the crowd" principle: while a single decision tree might be biased or prone to errors, the collective vote of hundreds of trees usually leads to a much more accurate and stable prediction.  

As such, many examples of applications and benefits of Random Forests in commerical settings are similar to those described in the Decision Tree project.  

The primary reason industries move from a single decision tree to a random forest is the trade-off between interpretability and performance.  A decision tree is a single flowchart-like structure, and hence easier to understand the logic.  In scenarios where an industry is highly regulated (like law or certain government sectors) and there is a need to explain exactly why every single decision was made, a single Decision Tree's transparency can be preferable, even if it produced lower accuracy.  The higher predictive accuracy of Randon Forests is typically the primary reason for using Random Forests.

The primary benefits of using Random Forests over Decision Trees include:

* Higher accuracy through ensemble voting
* Reduced overfitting via randomisation
* More robust to outliers and noise
* Better generalisation to unseen data

The key benefits of using a Random Forest include:

* Handles Missing Data: In the real world, data is rarely perfect. Random Forests can maintain high accuracy even when a significant portion of the data is missing.
* Feature Importance: It tells you which variables actually matter. For a business, knowing that "Customer Age" is 10x more important than "Postal Code" for sales is invaluable for strategy.
* No Need for Scaling: Unlike other models (like Neural Networks), you don't need to normalise your data (e.g., converting all numbers to a 0–1 scale). It works with raw numbers and categories out of the box.
* Parallelisation: Because each tree is built independently, they can be trained simultaneously on modern multi-core computers, making it very fast to train on large datasets.

## Methodology:  

The dataset used is the same as used in the Decision Tree project - the Wisconsin Breast Cancer dataset, which enables comparison of the two methods.  This is available from scikit-learn, including 569 observations, including 30 independent features.

The dataset is also available from Kaggle [here](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

The method applied in the analysis:

* **Dataset validation** to confirm no missing values, and basic descriptive analysis on the features including the correlation between the 30 features. No data pre-processing was undertaken.
* **Decision Tree Number Analysis** to determine the optimal number of trees in the forest, balancing accuracy of the model and compute resources required.
* **Decision Tree Depth Analysis** to determine the maximum depth of each tree in the forest to achieve optimal accuracy, prevent overfitting and balance with the compute resources required.

Details of the methodology applied in the project.

## Results and conclusions:

Results from the project related to the business objective.

Simple descriptive analytics determined that 212 observations relate to malignant cancers and 357 relate to benign cancers.

The correlation matrix is the same as that shown for the Decision Tree project [here](https://marcgrover-datascience.github.io/decision-trees/) as it uses the same dataset, and as such not shown here.

### Tree Number Analysis

When tuning hyperparameters like the number of trees (n_estimators) and tree depth (max_depth), the standard metric to apply is the Cross-Validation (CV) accuracy score based on the training set.

The alternative, using the accuracy score based on the test set to make these decisions, can lead to overly optimistic results and poor performance on truly "unseen" data.

Analysis was undertaken for Random Forests with the following number of trees; 10, 25, 50, 75, 100, 150, 200.  For each random forest the following metrics were calculated:

* accuracy on the training set
* accuracy on the test set
* Cross-Validation (CV) Accuracy score, where the number of folds was set to 5.

![tree_number](rf_trees_analysis.png)

### Tree Depth Analysis

![tree_depth](rf_depth_analysis.png)

![tree_example](rf_single_tree_structure.png)

![feature_importance](rf_feature_importance.png)


### Conclusions:

Conclusions from the project findings and results.

## Next steps:  

With any analysis it is important to assess how the model and application of the analytical methods can be used and evolved to support the business goals and business decisions and yield tangible benefits.


## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/RandomForest_BreastCancer.py)

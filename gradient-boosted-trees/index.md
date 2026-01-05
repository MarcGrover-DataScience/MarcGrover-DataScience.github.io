---

layout: default

title: Breast Cancer Predictions (Gradient Boosted Trees)

permalink: /gradient-boosted-trees/

---

#### This project is in development

## Goals and objectives:

The business objective is to predict the cancer status of cells (benign or malignant) based on 30 features of the cells observed via digitised images. Two previous projects built a decision tree model and random forest model for this prediction, achieving an accuracy of 93.86% and 95.61% respectively.  The goal of this project is to research if using a Gradient Boosted Trees predictor can produce more accurate results, and produce more insights into the data supporting the predictions. The predicted results are binary ‘Malignant’, ‘Benign’ values.

This project contains many references to the Decision Tree and Random Forest projects as one of the key analysis goals is to understand the comparison between the three approaches.

This follows on from the Decision Tree project found here , and the Random Forest project found here.

An optimal gradient boost trees model was built to make the predictions, achieving an accuracy of XX.XX%, using optimal hyperparameters xxxxx. For the random forest the most important feature was identified as ‘worse area’. The accuracy increased by X.XX% from the optimal random forest in the previous project.

## Application:  

There are multiple implementations of Gradient Boosted Trees, all use the same fundamental principle: sequentially building decision trees where each new tree corrects errors made by previous trees. However, they differ significantly in implementation details, speed, and specialisations.

For this proof-of-concept XGBoost is used.  XGBoost is conceptually similar to Random Forest but with boosting instead of bagging, making it an logical next step from random forests.  Other common examples of Gradient Boosted Trees (not used in this project) include LightGBM and CatBoost.  

As such, many examples of applications and benefits of Random Forests in commercial settings are similar to those described in the Decision Tree project. 

Details of how this is applicable to multiple industries to solve business problems, generate insight and provide tangible business benefits. 


## Methodology:  

Details of the methodology applied in the project.

## Results and conclusions:

Results from the project related to the business objective.

### Conclusions:

Conclusions from the project findings and results.

Decision Tree → Random Forest → XGBoost provides excellent narrative for showing evolution of ensemble methods.  

## Next steps:  

With any analysis it is important to assess how the model and application of the analytical methods can be used and evolved to support the business goals and business decisions and yield tangible benefits.


## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/DecisionTree_BreastCancer.py)

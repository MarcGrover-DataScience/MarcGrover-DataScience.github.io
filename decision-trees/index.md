---

layout: default

title: Breast Cancer Predictions (Decision Trees)

permalink: /decision-trees/

---

#### This project is in development

## Goals and objectives:

The business objective is to predict the cancer status of cells (benign or malignant) based on 30 features of the cells observed via digitised images.  A decision tree model was built to make the predictions, achieving an accuracy of 93.86%.

## Application:  

Details of how this is applicable to multiple industries to solve business problems, generate insight and provide tangible business benefits. 

Decision trees are powerful analytical tools that utilise a flowchart-like structure to classify data or predict outcomes by recursively splitting a dataset into smaller subsets based on specific feature criteria. Their primary appeal lies in their high interpretability, as they act as "white-box" models where the logic behind every conclusion is visually traceable and easy to explain to non-technical stakeholders. Beyond clarity, these models are exceptionally robust and versatile; they require minimal data preprocessing—meaning they don't need data scaling or normalization—and they naturally handle a mix of categorical and numerical variables, making them an efficient and accessible tool for solving complex logic-based problems across various industries.  

They are highly valued because they translate complex data into a visual, human-readable format that simplifies high-stakes decision-making.  

* **Finance** - decision trees are essential for managing risk and ensuring regulatory compliance through transparent logic, providing a clear "audit trail" of a reason for a decision.
  * Credit Scoring & Loan Approvals: Banks use decision trees to evaluate the creditworthiness of applicants. By inputting variables like income, debt-to-income ratio, and payment history, the tree classifies applicants into "High Risk" or "Low Risk" categories.  
  * Fraud Detection: Real-time transaction monitoring systems use decision trees to flag suspicious activity. For instance, if a transaction occurs in a new location for an unusually high amount at an odd time, the tree triggers an immediate alert or hold.
  * Option Pricing: Investors use "binomial trees" to estimate the value of financial options over time, helping them decide whether to buy or sell based on market volatility.  
* **Technology** - decision trees used to handle massive datasets and automate customer-facing processes.  Decision trees can make predictions or classifications almost instantly, which is vital for real-time web applications.  
  * Customer Churn Prediction: SaaS companies analyse usage patterns (e.g., login frequency, feature adoption) to identify customers at risk of cancelling. The tree helps pinpoint which specific behaviours are the strongest indicators of churn.  
  * Recommendation Engines: Streaming services and E-commerce platforms use tree-based models (often expanded into "Random Forests") to suggest products or movies based on a user's previous clicks and demographic data.  
* **Science & Healthcare** - decision trees help navigate complex biological and environmental variables to reach accurate conclusions.  Decision trees highlight which variables (e.g., which specific gene or symptom) are the most significant drivers of the outcome.  
  * Medical Diagnosis: Doctors use clinical decision trees to rule out conditions. For example, a tree for chest pain might branch into "History of Heart Disease" vs. "No History," further splitting by blood pressure and EKG results to reach a diagnosis.  
  * Genomic Research: Scientists use trees to classify sequences of DNA or proteins, identifying which genetic markers are most likely associated with specific diseases or traits.  
  * Environmental Modeling: Researchers use them to predict the impact of climate variables (like temperature and humidity) on crop yields or the spread of invasive species.  
* **Manufacturing** - decision trees are critical for maintaining high quality and optimising the flow of goods.  This can reduce costs, downtime and reputational damage and increase efficiencies.
  * Root Cause Analysis (RCA): When a batch of products fails quality testing, a decision tree helps technicians trace the defect, identifiying the factors most likely to be the cause, and help determine the exact point of failure.
  * Predictive Maintenance: Sensors on factory equipment feed data into trees that predict when a machine is likely to break down, allowing for repairs before an expensive halt in production occurs.
  * Supply Chain Optimisation: Logistics managers use trees to decide the best shipping routes or vendor selections based on lead times, costs, and historical reliability.

## Methodology:  

Details of the methodology applied in the project.

The dataset used is the Wisconsin Breast Cancer dataset available from scikit-learn, which contains 569 observations, including 30 independent features.

The dataset is also available from Kaggle [here](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

The dataset was validated to confirm that there are no missing values, and basic descriptive analysis was undertaken on the features including the correlation between the 30 features.  No data pre-processing was undertaken.

Decision tree depth analysis was undertaken to determine the optimal depth of the decision tree being created, to generate the most accurate model, and not cause overfitting.

## Results and conclusions:

Simple descriptive analytics determined that 212 observations relate to malignant cancers and 357 relate to benign cancers.

### Feature Correlation:  

Correlation of the 30 features was undertaken and visualised as a correlation matrix as shown below.  This highlights that many of the fields have low-correlation, however there appears to be high-correlation in the features relating to radius, area and perimeter metrics.  This was not addressed at this stage, but important insight for any future development to improve the predictions.

![correlation](correlation_matrix.png)

### Tree Depth Analysis:  

Training and testing sets were determined from the 569 observations in the data, where 80% of the data was for training, and the remaining 20% for testing.  For reference the training set included 455 samples of which 285 were benign cancers and 170 malignant.

Decision tree depth analysis was undertaken on levels in the range (1,13), for each level three metrics were calculated:
* accuracy on the training set
* accuracy on the test set
* Cross-Validation (CV) Accuracy score, where the number of folds was set to 5.  

The plot below shows the results of the tree depth analysis, which determined that a depth of 3 is optimal, however a depth of 4 also produced similarly accurate results.  This plot also showed that decision trees of 5 or more levels produced less accurate predictions, almost certainly due to over-fitting to the training data.  It is important that a decision tree is fitted with the optimum levels to generate the most accurate model.

![depth_analysis](depth_analysis.png)

### Model Fitting and Validation:

Using the tree depth of 3, the decision tree was trained, as visualised below. 

![decision_tree](decision_tree_structure.png)

The model performance was evaluated to quantify the quality of the predictions.  The key metics (based on the testing set) are:  
* Accuracy:  0.9386
* Precision: 0.9452
* Recall:    0.9583
* F1-Score:  0.9517

### Feature Importance:

![feature_importance](feature_importance.png)

### Conclusions:

Conclusions from the project findings and results.

## Next steps:  

With any analysis it is important to assess how the model and application of the analytical methods can be used and evolved to support the business goals and business decisions and yield tangible benefits.

* Feature engineering - including removing and / or adding features


## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/DecisionTree_BreastCancer.py)

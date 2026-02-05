---

layout: default

title: Project (Logistic Regression)

permalink: /logistic-regression/

---

# This project is in development

## Goals and objectives:

The business objective is to predict the churn of bank customers.  A Logistic Regression model is to be built using customer features and historical data to predict which current customers are likely to leave, which can be used to target actions to retain those customers.

https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
opportunities to demonstrate handling of imbalanced classes, which is common in fraud and default prediction scenarios where the positive class (default) is much rarer than the negative class.
Numerical or categorical data?

## Application:  

Details of how this is applicable to multiple industries to solve business problems, generate insight and provide tangible business benefits. 

**Logistic regression** is a fundamental statistical method used for classification tasks. At its core, logistic regression predicts the probability that a given input belongs to a particular class by applying the logistic (sigmoid) function to a linear combination of input features. The sigmoid function maps any real-valued number into a value between 0 and 1, making it ideal for probability estimation. This output can be interpreted as the likelihood of an observation belonging to the class, with a decision threshold (typically 0.5) used to make the final classification.  

For **binary classification** problems, the model learns weights for each feature that indicate how strongly that feature influences the probability of belonging to one class versus the other.  For example, in a medical diagnosis scenario predicting whether a patient has a disease, logistic regression might learn that certain symptoms or test results increase the probability of a positive diagnosis.  The model outputs a probability score, and any observation with a probability above the threshold is classified as the positive class, while those below are classified as negative.  This probabilistic nature is advantageous because it not only provides a prediction but also quantifies the model's confidence in that prediction.  

Logistic regression can be extended to handle **multi-class classification** problems through two primary approaches: **One-vs-Rest (OvR)** and **multinomial logistic regression** (also called softmax regression). 
* **One-vs-Rest** - separate binary logistic regression models are trained for each class, where each model distinguishes one class from all others combined. During prediction, all models generate probability scores, and the class with the highest probability is selected.
* **Multinomial logistic regression** -  generalises the binary case more naturally by using the softmax function to model the probability distribution across all classes simultaneously. This approach ensures that the predicted probabilities for all classes sum to 1, making it particularly suitable for classifying to multiple mutually exclusive categories (e.g. digit recognition, document categorisation, or iris species classification).

Logistic regression can be applied in a variety of scenarios across all industry sectors.  Example uses include:

* **Finance: Credit Default Prediction** - logistic regression models are extensively used for credit risk assessment and loan default prediction, to evaluate whether a loan applicant is likely to default on their payments.  The model takes into account various features such as credit score, income level, employment history, debt-to-income ratio, previous loan repayment behavior, and loan amount.  By analysing historical data from thousands of past borrowers, the model learns which combinations of these features are associated with higher default risk.  The output is a probability score between 0 and 1, representing the likelihood of default, supporting decisions on offering loans and setting interest rates.  This probabilistic approach supports data-driven lending decisions and risk exposure management, maintaining profitability while still serving customers.  
* **Medical: Disease Diagnosis and Screening** - logistic regression plays a crucial role in diagnostic decision support systems, particularly for screening and early detection of diseases.  For example in predicting the likelihood of heart disease based on patient characteristics and clinical measurements.  Medical practitioners input patient data including age, blood pressure, cholesterol levels, blood sugar levels, smoking status, family history, and results from tests like electrocardiograms.  The logistic regression model, trained on extensive patient databases, outputs a probability that the patient has or will develop heart disease.  This probability helps physicians prioritise which patients need immediate intervention, further diagnostic testing, or preventive care. The interpretability of logistic regression is particularly valuable in medical settings because doctors can understand which factors are driving the prediction, allowing them to explain risks to patients and make informed clinical decisions that combine the model's output with their professional expertise.  
* **Retail: Customer Churn Prediction** - Retail companies, especially those with subscription-based models or loyalty programs, use logistic regression to predict customer churn, which is the likelihood that a customer will stop doing business with them.  The model analyses customer behaviour patterns including purchase frequency, recency of last purchase, average transaction value, etc.  By identifying customers with high churn probabilities, retailers can proactively implement retention strategies such as personalised offers or targeted discounts.  
* **Manufacturing: Quality Control and Defect Detection** - In manufacturing, logistic regression is applied to predictive quality control, helping identify whether products will meet quality standards or be defective.  Production facilities collect vast amounts of data from sensors and inspection points throughout the manufacturing process. A logistic regression model trained on this data can predict the probability that a product being defective based on the process parameters observed during its production. Products can be identified as high-risk, enabling additional focussed additional checks or handled differently.  This can reduce waste, improve overall product quality, minimise risk and exposure, and optimise efficiency and profitability.  


## Methodology:  

Details of the methodology applied in the project.

The dataset used is publically available from Kaggle [here](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset).  The dataset contains 10,000 records which were used to train and test the Logistic Regression model.

## Results and conclusions:

Results from the project related to the business objective.

### Conclusions:

Conclusions from the project findings and results.

## Next steps:  

With any analysis it is important to assess how the model and application of the analytical methods can be used and evolved to support the business goals and business decisions and yield tangible benefits.


## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/x.py)

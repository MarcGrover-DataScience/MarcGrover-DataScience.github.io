---

layout: default

title: Restaurant tips (Multiple Linear Regression)

permalink: /multi-linear-regression/

---

## Goals and objectives:

The business objective is to determine how accurately tip amount can be predicted from three observable variables — total bill value, party size, and time of day — and to understand the relative contribution of each predictor to that outcome. Multiple Linear Regression (MLR) was applied to 244 observations from the Seaborn Tips dataset, providing a controlled, reproducible setting in which to demonstrate a complete regression analysis pipeline.

The analytical scope goes beyond model fitting. The project applies statistical significance testing via a statsmodels OLS model, producing p-values, t-statistics, an F-statistic, adjusted R², and 95% confidence intervals for each coefficient — establishing not just whether the model predicts, but whether each individual predictor contributes meaningfully. Assumption testing is conducted formally using the Shapiro-Wilk test for residual normality and the Spearman Correlation test for homoscedasticity, with multicollinearity assessed through Variance Inflation Factors (VIF). Influential observations are identified using Cook's Distance. Model performance is evaluated both on a held-out test set and through 10-fold cross-validation, providing a more robust estimate of generalisation than a single split allows on a dataset of this size.

A tip percentage analysis is included to contextualise the heteroscedasticity finding: examining tip as a proportion of bill value reveals why variance grows with bill size, and motivates the application of a square-root transformation to the target variable. A second model using √tip (square-root of tips values) is fitted and evaluated in direct comparison with the original, with the Spearman test re-run to assess whether the transformation resolves the heteroscedastic pattern.

The project confirms that total bill amount is the strongest and statistically significant predictor of tip size, with the transformed model providing a modest improvement in explained variance. Key limitations — including moderate multicollinearity between total bill and party size, and the absence of service quality or customer satisfaction data — are identified alongside recommendations for further development.

## Application:  

The business application of MLR is centred on forecasting (predicting future values) and causal analysis (quantifying the impact of multiple factors on a single outcome).  Examples of uses across different sectors include (though this is a far from complete list)

🏦 In **finance**, MLR is essential for risk modelling, asset valuation, financial forecasting, and understanding market drivers.  
🛍️ **Retail** leverages MLR primarily for sophisticated demand forecasting, pricing strategies, customer life valuations and marketing budget allocation.  
💻 In **technology**, MLR is vital for optimizing product performance, user retention, and managing infrastructure costs.  
🏭 In **manufacturing** MLR is fundamental to Quality Control (QC) and Predictive Maintenance by modelling the complex interplay of process variables.  

## Methodology:  

A workflow in Python was developed using libraries Scikit-learn, Scipy, Pandas and Numpy, utilising Matplotlib and Seaborn for visualisations.  The data used was the Tips dataset from within Seaborn.  

A Multiple Linear Regression model was built, having processed and scaled the independent variable datasets.  Tests and analysis were performed on the data for:
* Correlation of variables
* Normality of the residuals (using Shapiro-Wilks)
* Homoscedasticity of the predictions and absolute residuals (using the Spearman Correlation test)
* Multicollinearity of the independent variables using Variance Inflation Factors (VIF)
* Feature Importance Ranking to determine the strongest predictors of tip value

Data preparation:  Minor transformation of data into a pandas dataframe and contingency table for analytical purposes.  Note that for analytical purposes, the time column, which stated the sitting as either lunch or dinner, was converted into an integer value where dinner is represented by 1 and lunch by 0.  Scaling of the factors was undertaken as part of building the MLR model. 

## Results and conclusions:

### Descriptive Statistics:

There are 244 observations recorded, covering a period of 4 consecutive days.

The mean tip is 2.998, with a standard deviation of 1.384 (currency is ignored in this example for simplicity)

The following scatter plot shows the mapping of tip value to total bill value:

![scatter_billing](mlr_scat_bill.png)

The following boxplot visualises the tip distribution for both lunch and dinner sittings:

![boxplot_time](mlr_boxplot_time.png)

### Correlation Analysis

The correlation matrix (for the 3 independent variables and dependent variable) shows the correlation between each variable.  

Values close to 1 or -1 indicate strong relationships, and values close to 0 indicate weak relationships.

![correlation_matrix](mlr_corr_mat.png)

### Multicollinearity test:

Multicollinearity needs to be tested in multiple linear regression because it can significantly distort the results and make the model unreliable and difficult to interpret.

Multicollinearity occurs when two or more independent variables (predictors) in a regression model are highly correlated with each other.

The most common method for detecting multicollinearity is by calculating the Variance Inflation Factor (VIF) for each independent variable.

Variance Inflation Factor (VIF):  

```
  Feature    VIF  
 total_bill  9.216  
       size  9.271  
time_dinner  3.170
```

The general guidance states that a VIF > 10 is considered 'High multicollinearity', which is not the case here, but with two values greater than 9.2 this should be noted.

### Hypothesis Test:  

The data was split into training and test sets using the standard 80/20 ratio.

Feature scaling was undertaken, where the independent variable (X) data was also scaled so within each IV the mean = 0 and standard deviation = 1.

Feature scaling is important, as the different features have different ranges (e.g., bill: $3-50, size: 1-6).  Scaling puts all features on the same scale, which makes the coefficients directly comparable, and also improves the model training and interpretation.

Note that before scaling, the mean and standard deviation for each IV were:  

```
        total_bill     size  time_dinner  
mean        20.218    2.574        0.728  
std          8.771    0.941        0.446
```

The MLR model was fitted with the scaled IV data, producing a model, based on scaled versions of the factors:

Tip = 3.088 + (0.801×total_bill) + (0.248×size)  - (0.024×time_dinner)

The model was applied to the test data so that there was both predicted and actual tips for those observations, enabling a calculation for the residual for each test observation (actual tip - predicted tip).

### Model Evaluation:

First the model was evaluated against the training set, returning: 
R² Score: 0.4519 (45.2% of variance explained by the model)  
Root Mean Squared Error: 1.06  
Mean Absolute Error: 0.76  

The model was then evaluated against the test set, returning:
R² Score: 0.4772 (47.7% of variance explained)  
Root Mean Squared Error: 0.81  
Mean Absolute Error: 0.67  

Using the evaluation of the test set, this means that our model explains 47.7% of the variation in tips, and using MAE, on average the absolute error is 0.67

The scatter plot below plots the predicted values against the actual values in the test set.  Points close to the red line, equate to good predictions, and points far from the red line relate to less accurate prediction errors.  The random scatter around the line further implies a good model fit.

![predictions_scatter](mlr_scatter_pred_act.png)

### Residual Analysis:

The residuals, where each Residual = Actual Value - Predicted Value , are plotted below to visualise in a different way the predictions against the actuals.  The points near the red-line represent accurate predictions.

The analysis of the residuals show that the mean = -0.2446 (where this should be close to 0), and the standard deviation = 0.78 (where a value close to zero represents a good model).

The residual plots look random, and without pattern, and also does not look funnel or cone shape, which implies equal variances (homoscedasticity).  These further confirm that the model is good, however further, more accurate, analysis is detailed in the next section.

![residuals_scatter](mlr_scatter_res.png)

![residuals_histogram](mlr_hist_res.png)

### Testing Assumptions:

Linear regression requires these assumptions to be tested:

#### Test 1 - Residual normality - where the null hypothesis is that residuals are normally distributed.

Using the Shapiro-Wilk Test the results returned a P-value: 0.4930, and as p > 0.05, this is evidence that the residuals are normally distributed as required.

#### Test 2 - Homoscedasticity (constant variance of residuals)

Using the Spearman Correlation test on the predicted values and the absolute residuals, the P-value was equal to 0.0000 , the Spearman Correlation value was 0.6063.

As such p < 0.05 this is evidence of heteroscedasticity (p ≤ 0.05), therefore we should consider transforming the target variable accordingly.

#### Test 3 - No multicollinearity among features

We already tested this with VIF, see above, where the result that there is moderate multicollinearity present.

### Feature importance:

Understanding the importance of each feature (Independent Variable) is an important finding from Multiple Linear Regression, as it allows further understanding of the model and output, as well as guiding any further improvements to be made to the model.

The bar chart below visualises the importance of each feature, showing that, perhaps unsuprisingly, total bill value is the strongest predictor of tip size

![feature importance](mlr_feat_imp.png)

### Conclusions:

Lets address the conclusions in relation to our research question:  Can we predict restaurant tips based on bill value, party size, and time?

Model Performance:
* The model works reasonably well for a simple dataset (47.7% of variance explained)
* The average prediction error is 0.67

Keyfindings:
* Total bill amount is the strongest predictor of tip size, the higher the bill amount the higher the tip tends to be
* Larger parties tend to leave larger tips
* Lunch time sittings decreases expected tips

Assumption testing:
* Normality: Satisfied
* Constant Variance: Some heteroscedasticity
* Multicollinearity: Moderate

Practical interpretation:
For every increase of 1 in the total bill, we expect the tip to increase by approximately 0.09, holding other factors constant.

Model and Analysis Limitations:
* Model doesn't account for service quality or customer satisfaction
* R² of 0.477 means 52.3% of variation is unexplained
* Limited to the patterns in this restaurant's data

## Next steps:  

With any analysis it is important to assess how the model and data collection can be improved to better support the business goals.

Recommendations include:
* Collect additional features (e.g., day of week, server ID, meal type, customer satisfaction)
* Address the moderate multicollinearity identified in the variables
* Increase sample size for better generalisation
* Consider non-linear relationships (polynomial features)
* Transform the dependent variable (to address the heteroscedasticity detected from the Spearman Correlation test)
  * Note that a separate model was generated in a second phase using the square-root of the tip value increased the model accuracy slighty to account for 48.2% of the variance)
* Explore interaction effects between features

## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/Multi-Linear-Regression-Tips_v2.py)

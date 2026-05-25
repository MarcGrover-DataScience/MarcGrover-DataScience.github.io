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

Multiple Linear Regression models a continuous outcome as a simultaneous function of two or more independent variables, producing a fitted equation that quantifies the independent contribution of each predictor — that is, its effect on the outcome holding all other variables constant. This dual capability — prediction and causal decomposition — distinguishes MLR from many alternatives. Unlike black-box models, MLR produces interpretable, statistically testable coefficients for each feature, enabling analysts to answer not just what an outcome will be, but which factors drive it and by how much. The formal statistical framework that surrounds it — significance testing of individual coefficients, F-statistic assessment of overall model fit, adjusted R² penalising redundant predictors, and assumption diagnostics including normality, homoscedasticity, and multicollinearity testing — makes MLR particularly well-suited to regulated, auditable, and decision-critical contexts where transparency and accountability are as important as predictive accuracy. It is routinely applied as a rigorous baseline against which more complex models are benchmarked.

The business application of MLR spans any domain where a continuous outcome is driven by multiple measurable factors. Practical examples showing where MLR provides clear analytical value include:

🏦 **Finance**:

**Mortgage and loan pricing**: Lenders model expected credit loss as a function of borrower characteristics — loan-to-value ratio, income, credit score, employment type — with MLR coefficients quantifying the independent risk contribution of each factor and providing a directly auditable basis for pricing decisions.

**Equity factor modelling**: Analysts regress asset returns against macroeconomic and market factors (market return, interest rate sensitivity, sector index) to estimate factor exposures and decompose portfolio risk — the analytical foundation of multifactor equity models.

**Insurance premium setting**: Actuaries quantify the independent contribution of age, vehicle type, claims history, and geography to expected claim cost, producing premium structures that are statistically defensible and regulatorily transparent.

🏠 **Real estate**:

**Hedonic property valuation**: MLR is the classical tool for decomposing property prices into the contribution of individual characteristics — floor area, number of bedrooms, distance to transport links, energy rating — with coefficients directly interpretable as the market value of each attribute.

**Rental yield forecasting**: Property investors and analysts model gross yield against location indices, occupancy rates, and property age to identify mispriced assets and support acquisition decisions.

🛍️ **Retail and marketing**:

**Marketing mix modelling**: MLR is the backbone of MMM — quantifying the independent return on investment of each marketing channel (TV, paid digital, in-store promotions) on sales volume, enabling budget reallocation decisions grounded in estimated marginal contribution rather than correlation.

**Price and promotional elasticity**: Retailers model sales volume as a function of own price, competitor price, and promotional intensity, using the fitted coefficients to simulate the revenue impact of pricing changes before implementation.

🏥 **Healthcare**:

**Clinical outcome prediction**: Hospital systems model patient length of stay or readmission probability against admission characteristics — age, diagnosis code, comorbidity count, admission route — to support capacity planning and target early-intervention resources at high-risk patients.

**Health economics and policy**: Epidemiologists quantify the independent contribution of lifestyle, socioeconomic, and demographic factors to health outcomes at population level, providing a statistically controlled evidence base for public health policy design.

🏭 **Manufacturing**:

**Process yield optimisation**: Quality engineers model defect rate or product yield as a function of controllable process parameters — temperature, pressure, line speed, operator shift — using the regression coefficients to identify which variables most influence output quality and define optimal operating ranges.

**Predictive maintenance scheduling**: Maintenance teams regress equipment degradation rate against operational metrics (vibration amplitude, temperature variance, run hours) to predict failure risk and schedule intervention before unplanned downtime occurs.

💻 **Technology**:

**Infrastructure cost modelling**: Engineering and finance teams regress cloud spend against product usage metrics — API call volume, active users, data storage consumption — to attribute costs accurately across services and forecast expenditure as the product scales.

**Product analytics and retention**: Product teams quantify the independent effect of feature adoption, session frequency, and support contact rate on user retention or NPS, identifying the levers most worth investing in and those that are statistically inert.

## Methodology:  

The analysis is implemented in Python using pandas for data handling, scikit-learn for modelling and preprocessing, statsmodels for statistical inference, scipy for assumption testing, and seaborn and matplotlib for visualisation. The dataset is the Tips dataset, loaded directly from seaborn, comprising 244 observations across seven variables.

**Data Loading and Preparation**

The dataset is confirmed to contain no missing values, requiring no imputation or record removal. Three independent variables are selected for the model: 
* total bill value
* party size
* time of day

The categorical time variable — recorded as 'Lunch' or 'Dinner' — is binary-encoded as an integer column (time_dinner), where Dinner = 1 and Lunch = 0. The remaining categorical variables (sex, smoker, day) are excluded from this model, with their potential contribution noted as an area for future development.

**Exploratory Data Analysis**

Descriptive statistics are calculated for all variables. Three charts are produced to examine the relationship between each independent variable and tip amount prior to modelling: a scatter plot of total bill against tip, a scatter plot of party size against tip with horizontal jitter applied to reduce overplotting on the integer-valued axis, and a boxplot comparing tip distributions across lunch and dinner sittings. A correlation matrix heatmap is produced to quantify pairwise linear relationships between all three predictors and the target variable.

A tip percentage analysis is conducted — expressing tip as a proportion of total bill for each observation — to examine whether tipping behaviour is proportional or additive across the bill range. This analysis directly contextualises the heteroscedasticity finding identified in the assumption testing stage.

**Multicollinearity Assessment**

Prior to model fitting, the Variance Inflation Factor (VIF) is calculated for each independent variable. A VIF exceeding 10 is treated as indicative of high multicollinearity; values between 5 and 10 are noted as moderate. Results are visualised as a bar chart with threshold reference lines.

**Model Specification and Fitting**

The dataset is partitioned into training and test sets using an 80/20 split with a fixed random state for reproducibility. Feature scaling is applied using scikit-learn's StandardScaler, fitted exclusively on the training set and applied to both, ensuring no data leakage between partitions. Two complementary model specifications are fitted:

* **scikit-learn LinearRegression** is fitted on the scaled training data, producing standardised coefficients used for feature importance comparison and generating predictions on both training and test sets.
* **statsmodels OLS** is fitted on the unscaled features with a constant term added, to obtain the full inferential output — p-values, t-statistics, model-level F-statistic, adjusted R², and 95% confidence intervals for each coefficient. This provides the statistical basis for assessing whether each predictor's contribution is meaningfully distinguishable from zero, which scikit-learn does not provide.

**Model Evaluation**

Performance is assessed on both training and test sets using R², Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE). To complement the single train/test split — which produces a test set of only 49 observations on a dataset of this size — 10-fold cross-validation is applied using a scikit-learn Pipeline that encapsulates scaling and model fitting within each fold, eliminating any risk of leakage across folds. Mean and standard deviation of R², MAE, and RMSE across folds are reported alongside the single-split results, providing a more reliable estimate of generalisation performance.

**Residual Analysis and Assumption Testing**

Residuals are examined visually through scatter plots of residuals against predicted values, histograms of the residual distribution, and Q-Q plots — each produced for both the training and test sets. Three formal assumption tests are then conducted:

* **Residual normality** is tested using the Shapiro-Wilk test, with H₀ that residuals are normally distributed, evaluated at α = 0.05.
* **Homoscedasticity** is assessed via the Spearman Correlation test between predicted values and absolute residuals; a statistically significant result indicates that variance is not constant across the prediction range.
* **Multicollinearity** is addressed through the VIF analysis conducted prior to model fitting.

Influential observations are identified using Cook's Distance, calculated via statsmodels' OLSInfluence. The conventional threshold of 4/n is applied to flag observations whose removal would materially alter the fitted model.

**Target Variable Transformation**

Where the Spearman test identifies heteroscedasticity in the original model, a square-root transformation is applied to the target variable and a second model is fitted using the same train/test split and scaling procedure. Predictions are back-transformed to the original tip scale for direct metric comparison. The Spearman test is re-run on the transformed model's residuals to determine whether the transformation resolves the heteroscedastic pattern, with results presented alongside a side-by-side residual plot comparison of both models.

## Results and conclusions:

### Descriptive Statistics:

The dataset comprises 244 observations collected across four consecutive days. The mean tip is 2.998 with a standard deviation of 1.384. The scatter plot below maps tip amount against total bill value, showing a positive linear trend with increasing spread at higher bill values — a pattern that anticipates the heteroscedasticity finding discussed in the assumption testing:

![scatter_billing](mlr_scat_bill.png)

The boxplot below compares the tip distribution across lunch and dinner sittings. Dinner sittings show a higher median tip and greater spread, though the overlap between groups is substantial:

![boxplot_time](mlr_boxplot_time.png)

### Correlation Analysis:

The correlation matrix quantifies the pairwise linear relationships between all variables. Total bill shows the strongest correlation with tip (r ≈ 0.68), party size shows a moderate positive correlation (r ≈ 0.49), and time of day shows a weaker relationship (r ≈ 0.27). The correlation between total bill and party size (r ≈ 0.60) is notably elevated and is examined further in the multicollinearity assessment:

![correlation_matrix](mlr_corr_mat.png)

### Tip Percentage Analysis:

Expressing tip as a percentage of total bill reveals the source of the heteroscedasticity detected in assumption testing. Tip percentage is most variable at lower bill values and stabilises at the higher end of the bill range, producing the fan-shaped spread visible in the earlier scatter plot. This confirms that tipping is not purely proportional — some customers tip a fixed amount regardless of bill size — and explains why residual variance is systematically higher for lower predicted values:

![mlr_tip_pct_vs_bill](mlr_tip_pct_vs_bill.png)

### Multicollinearity Assessment:

VIF values were calculated for each independent variable prior to model fitting:

```
  Feature    VIF  
 total_bill  9.216  
       size  9.271  
time_dinner  3.170
```

No variable exceeds the high-multicollinearity threshold of 10, however the values for total bill (9.216) and party size (9.271) are close to that boundary, reflecting the moderate correlation between these two predictors (r ≈ 0.60). The independent coefficient estimates for these features carry more uncertainty than they would in a fully orthogonal design and should be interpreted accordingly.

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

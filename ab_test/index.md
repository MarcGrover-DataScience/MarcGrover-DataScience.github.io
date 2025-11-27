---

layout: default

title: Web conversion rates  (A/B test)

permalink: /ab-test/

---

## Goals and objectives:

To test and analyse the conversion rates for users signing up for membership via a new webpage (treatment group) compared to the old webpage (control group).  The business wants to understand if there are any statistically significant differences.

1,000 data points were collected from both the new and old webpage, which showed 180 took out membership via the new web page, and 120 took out membership via the old web page.  This data is analysed to determine if any significant difference can be assumed, and identify other insight from the data.

The A/B test (using the chi-squared test) provided evidence that there is a statistically significant difference in conversions between the two versions of the website, and provided the business with evidence to support the decision to migrate to the new version of the website.  (Explain the expected benefits)

## Application:  

Why use the chi-squared test? The chi-squared t-test and the 2-sample test can both be used in A/B testing, but they are appropriate for different types of metrics and research questions.  The best test depends entirely on the type of data you are analysing.  

The chi-squared test of independence is used to compare the proportions or frequencies of a categorical variable between two groups.  As such it is applicable to cases such as this project where the data categorical/discrete, i.e. testing if the conversion rate (proportion of users who converted) is significantly different between the Control (A) and Variant (B).

By contrast, the independent 2-sample t-test is used to compare the means (averages) of a continuous variable between two independent groups (A and B).

A/B testing used to compare proportions or frequencies of a categorical variable is foundational to modern data-driven decision-making, particularly in optimizing digital experiences and quality control.

* In the technology sector, this is the most common use of A/B testing, focusing on user behavior that results in a binary outcome (success/failure).  Examples include, conversion, click-through rates, email-open rates.
* Retail uses these tests to optimize both online and in-store campaign effectiveness - offer redemption rates, cart abandonment rate, packing preference.
* In finance, these tests are crucial for improving the efficiency of lead generation and customer onboarding - Application Submission Rate, Lead-to-Client Conversion.
* In manufacturing, this A/B testing framework is used offline to compare the effectiveness of two production conditions on a binary quality outcome - Defect Rate Comparison, Pass/Fail Inspection Rates.

## Methodology:  

A workflow in Python was developed using libraries Scipy, Pandas and Numpy, utilising Matplotlib for visualisations.  The data was created in the script, with the intention of producing interesting statistical findings.  

The A/B test was used to test the null hypothesis that there is no variance between the control and treatment groups.  Further analysis determined the high-confidence range of conversion percentages, to support business planning and expectations.

Tests were also undertaken to determine if the sample size was sufficient to detect a real difference given the expected conversion rates.

Data preparation:  Minor transformation of data into a pandas dataframe and contingency table for analytical purposes.

## Results and conclusions:

### Descriptive statistics:  

The data being used for the A/B test contains 1,000 data points for each group, where 18% converted in the treatment group and 12% converted in the control group.

![conversion](ab_conversion.png)

This data was used to create the contingency table:

![cont_table](ab_cont_tab.png)

The chi-squared test was applied to the data, with the null hypothesis that there is no variance between the control and treatment groups, with the significance level (alpha) equal to 0.05.  
The result was a p_value of 0.00022, and as this is > 0.05 we can reject the null hypothesis and have evidence that there is a statistically significant difference in conversion rates between the 2 groups.

![exp_freq](ab_exp_freq.png)
![conversion_ci](ab_conversion_ci.png)


Two-Way ANOVA without replication test is:



Initially boxplots of alcohol content by each factor are created:  



A heatmap of the values is also produced as provides a useful visualisation of the data:  



An interactive plot also enhances understanding of the data: 



An initial visual inspection of these charts can lead to the assumption that the factors have a strong relationship to the alcohol content, but lets test that hypothesis and understand more about any relationships. 

### Hypothesis Test:

First we test the data for normality, using the Shapiro-Wilk Normality Test:  

Test Statistic: 0.921433  
P-value: 0.404236  
As the p_value > 0.05 - we can assume that the data is normally distributed  

Another assumption that we need to test is for homogeneity of variances, using Levene's Test for Equal Variances for each factor individually, the results being:

Levene's Test (by Quality):  
Test Statistic: 0.396240  
P-value: 0.689237  
As the p_value > 0.05 - Equal variances assumed

Levene's Test (by pH Level):  
Test Statistic: 0.460375  
P-value: 0.651620  
As the p_value > 0.05 - Equal variances assumed  

As such it can be considered that there is homogeneity of variances.

The Two-Way ANOVA without replication test was applied to the data for the two factors, where the alpha was set to 0.05 - i.e. 95% confidence, and the null hypothesis being that the two factors (quality and pH), do not have an effect on the dependent variable (alcohol content). The results for each factor were: 

MAIN EFFECT: QUALITY  
F-statistic: 22.4012  
P-value: 0.006718  - compared to the alpha = 0.05, i.e. p_value < 0.05  
 
Wine quality has a statistically significant effect on alcohol content, and we reject the null hypothesis that quality levels have equal mean alcohol content.

MAIN EFFECT: pH LEVEL  
F-statistic: 14.3968  
P-value: 0.014878  - compared to the alpha = 0.05, i.e. p_value < 0.05  

pH level has a statistically significant effect on alcohol content, we reject the null hypothesis that pH levels have equal mean alcohol content.

So in combination we can conclude that both wine quality and pH level have a statistically significant effect on alcohol content.  

Assessing the overall model the R-squared = 0.9485, which can be interpreted as 94.85% of the variance in alcohol content is explained by the two factors (Quality and pH Level) combined.  The p-value of the overall model is 0.007698, which is less than the aplha = 0.05, therefore we can reject the hypothesis that the model isn't any good, i.e. we can conclude that the model is good.

We need next to assess the effect sizes of each factor, where the results of effect (η² - Eta-squared) are, which are interpretted using Cohen's D values for effect size:

Quality effect size (η²): 0.5774 - i.e. 57.7% of the variance of alcohol content can be explained by the quality  
Cohen's D: Large effect

pH Level effect size (η²): 0.3711 - i.e. 37.1% of the variance of alcohol content can be explained by the pH level  
Cohen's D: Large effect

In summary the conclusions are that:
 - The model explains 94.85% of variance in alcohol content
 - Both factors should be considered when analyzing wine characteristics
 - Results suggest that wine quality and pH chemistry relate to alcohol levels

It should be noted that there are important limitations with this test:
 - this is considered an observational study, and as such causation cannot be inferred.  
 - sample represents single values (no replication within cells)
 - Results specific to the limited samples of wine

## Next steps:
Given the findings and limitations, and the limited number of measurements, it would be recommended to take more measurements for each factor combination, and potentially increasing the analysis to include more factors.  Such data should be subjected to other analytical methods, such as 2-way ANOVA with replication.  This may highlight interaction effects between factors.

## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/AB Testing.py)

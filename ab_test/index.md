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

The Two-Way ANOVA Without Replication (also known as a Randomized Block Design or a two-factor ANOVA with one observation per cell) is a statistical test used to assess the main effects of two categorical factors on a single continuous dependent variable, while assuming there is no interaction between the two factors.  This design is often used when one factor is a "nuisance factor" or a blocking variable that is included primarily to account for variability, thus increasing the power of the test for the other factor.

* Used in manufacturing where resources (materials, machines, time) are limited, and one observation per condition is practical for quality audits.
* In development, it's used to efficiently evaluate the main performance effects of two variables (e.g. software versions and server regions)
* Retail often uses this to compare key performance indicators (KPIs) across stores or channels, e.g. sales based on factors of store locations and promotional offers.
* In finance, it's used for comparative analysis where confounding variables need to be controlled (e.g. investment strategies and time periods)

## Methodology:  

A workflow in Python was developed using libraries Scipy, statsmodels, scikit-learn, Pandas and Numpy, utilising Matplotlib and Seaborn for visualisations.  The data came from a publicly available dataset of wine data from the library scikit-learn, which was then extended to produce interesting statistical findings.  

The Two-way ANOVA without replication test was used to test the null hypothesis that the two factors (quality and pH), do not have an effect on the dependent variable (alcohol content).  

Assumptions are tested regarding both the normality of the residuals using the Shapiro-Wilk test, and the homogeneity of Variances (Homoscedasticity) using Levene's test

The assumption of independence of observations is assumed due to design of the experiment, as well as the assumption of additivity or no interaction between the two independent factors..

Data preparation:  Minor transformation of data into a pandas dataframe for analytical purposes, where there is a single value of alcohol content for each quaility and pH combination.

## Results and conclusions:

### Descriptive statistics:  

The data being used for the Two-Way ANOVA without replication test is:

![input_data](2w_anova_without_ph_df.png)

Initially boxplots of alcohol content by each factor are created:  

![boxplot_by_ph](2w_anova_without_ph_boxplot.png)
![boxplot_by_quality](2w_anova_without_qual_boxplot.png)

A heatmap of the values is also produced as provides a useful visualisation of the data:  

![heatmap](2w_anova_without_ph_heat.png)

An interactive plot also enhances understanding of the data: 

![interaction_plot](2w_anova_without_interaction.png)

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
[View the Python Script](/ANOVA_2-way_withoutRep.py)

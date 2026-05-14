---

layout: default

title: Wine quality and alcohol content (Two-Way ANOVA without Replication)

permalink: /2-way-anova-without-rep/

---

## Goals and objectives:

The business objective is to test and analyse the relationship between 2 independent variables, wine quality ratings (5, 6, 7) and pH levels (Low, Medium, High), on alcohol content in wine, where there is a single measure of alcohol content for each combination of quality and pH level - hence there being no repetition of dependent variable values.  The business wants understand the relationship between quality and pH level on the alcohol content to guide further development, research and new product strategy.    

The Two-Way ANOVA without replication test provided evidence that the model explains 95% of variance of alcohol content using both factors (quality and pH level), and that both factors should be considered when analysing wine characteristics.

## Application:  

The Two-Way ANOVA Without Replication (also known as a Randomized Block Design or a two-factor ANOVA with one observation per cell) is a statistical test used to assess the main effects of two categorical factors on a single continuous dependent variable, while assuming there is no interaction between the two factors.  This design is often used when one factor is a "nuisance factor" or a blocking variable that is included primarily to account for variability, thus increasing the power of the test for the other factor.

* Used in manufacturing where resources (materials, machines, time) are limited, and one observation per condition is practical for quality audits.
* In development, it's used to efficiently evaluate the main performance effects of two variables (e.g. software versions and server regions).
* Retail often uses this to compare key performance indicators (KPIs) across stores or channels, e.g. sales based on factors of store locations and promotional offers.
* In finance, it's used for comparative analysis where confounding variables need to be controlled (e.g. investment strategies and time periods).

## Methodology:  

A workflow in Python was developed using libraries Scipy, statsmodels, scikit-learn, Pandas and Numpy, utilising Matplotlib and Seaborn for visualisations.  The data came from a publicly available dataset of wine data from the library scikit-learn.  

### Data Loading and Validation:

The dataset is sourced from the UCI Wine Quality (Red) dataset via scikit-learn's fetch_openml() function. The raw dataset contains 1,599 observations of red wine, each described by eleven physicochemical measurements and a quality rating assigned by sensory panels on an integer scale of three to eight. A data validation audit was conducted prior to analysis, confirming no missing values across any variable and verifying that all columns carry the expected numeric data types. Quality ratings were filtered to include only scores of 5, 6, and 7 — the three most represented ratings in the dataset and the basis for a balanced factorial design. pH values were discretised into three equal-width bands (Low, Medium, High) using pd.cut(), producing a 3×3 design matrix. Alcohol content was then averaged within each quality-pH combination, yielding nine unique cells — one observation per factor combination — as required by the no-replication design.

### Hyothesis Test:

The Two-way ANOVA without replication test was used to test the null hypothesis that the two factors (quality and pH), do not have an effect on the dependent variable (alcohol content).  

Assumptions are tested regarding both the normality of the residuals using the Shapiro-Wilk test, and the homogeneity of Variances (Homoscedasticity) using Levene's test.

The assumption of independence of observations is assumed due to design of the experiment.

The additivity assumption — that no interaction exists between quality and pH level — is a critical design requirement of the no-replication framework and is assessed visually via the interaction plot in the Results section.

Data preparation:  Minor transformation of data into a pandas dataframe for analytical purposes, where there is a single value of alcohol content for each quality and pH combination.

## Results:

### Descriptive statistics:  

The data being used for the Two-Way ANOVA without replication test is:

```
 Quality  pH_Level  Alcohol
       5       Low    9.743
       5    Medium    9.909
       5      High   10.548
       6       Low   10.326
       6    Medium   10.636
       6      High   11.623
       7       Low   11.029
       7    Medium   11.511
       7      High   13.100
```

Boxplots of alcohol content by each factor were created:  

![boxplot_by_ph](2w_anova_without_ph_boxplot.png)
![boxplot_by_quality](2w_anova_without_qual_boxplot.png)

A heatmap of the values is also produced as provides a useful visualisation of the data:  

![heatmap](2w_anova_without_ph_heat.png)

An interaction plot is produced to visually assess the relationship between the two factors across levels of alcohol content:

![interaction_plot](2w_anova_without_interaction.png)

In a Two-Way ANOVA without replication, the interaction plot serves a purpose beyond general data exploration — it provides the primary visual check of the additivity assumption, which is the central and non-negotiable design assumption of this test. Because the no-replication design has only one observation per cell, there are no degrees of freedom remaining to estimate an interaction term between the two factors. The residual variance in the model is the interaction variance. This means that if a true interaction exists between wine quality and pH level — that is, if the effect of pH on alcohol content genuinely differs depending on quality rating — the model has no mechanism to detect or account for it. That interaction variance would instead be absorbed into the error term, inflating the F-statistics for both main effects and potentially producing false positives. The additivity assumption must therefore be evaluated before the ANOVA results can be trusted.

Visually, the additivity assumption is supported when the lines on the interaction plot are approximately parallel. Parallel lines indicate that the difference in alcohol content between pH levels is consistent across all quality ratings — in other words, that the two factors act independently and additively on the outcome, with no meaningful interaction between them.
Inspecting the plot, the three lines — one per pH level — are broadly parallel across the quality ratings of 5, 6, and 7. All three lines rise from left to right, and the vertical spacing between them remains reasonably consistent across quality levels. There is no pronounced crossing or convergence of lines that would suggest the effect of pH on alcohol content is materially different at one quality rating compared to another. This provides visual support for the additivity assumption and gives reasonable confidence that proceeding with the no-replication ANOVA is appropriate for this data.

It is worth noting that visual inspection alone is not a formal test of additivity. Tukey's one-degree-of-freedom test for non-additivity is the standard formal procedure for this purpose and would be the recommended next step in a more rigorous analysis — particularly given the small number of cells in this design, where visual judgement is inherently limited. This is discussed further in the Next Steps section.

### Hypothesis Test:

An assumption that we need to test is for homogeneity of variances, using Levene's Test for Equal Variances for each factor individually, where the null hypothesis is that the variances are the same. 

Levene's test is included for methodological completeness but the sample size per group is too small for it to be truely informative, and the assumption should instead be defended on the basis of the experimental design and prior knowledge.  The results are below, suggesting there is homogeneity of variances.

Levene's Test (by Quality):  
Test Statistic: 0.396240  
P-value: 0.689237  
As the p_value > 0.05 - Equal variances assumed

Levene's Test (by pH Level):  
Test Statistic: 0.460375  
P-value: 0.651620  
As the p_value > 0.05 - Equal variances assumed  

The Two-Way ANOVA without replication test was applied to the data for the two factors, where the significance (alpha) was set to 0.05 - i.e. 95% confidence level, and the null hypothesis being that the two factors (quality and pH), do not have an effect on the dependent variable (alcohol content). The results for each factor were: 

MAIN EFFECT: QUALITY  
F-statistic: 22.4012  
P-value: 0.006718  - compared to the alpha = 0.05, i.e. p_value < 0.05  
 
This evidence supports the alternate hypothesis that the wine quality has a statistically significant effect on alcohol content, and we reject the null hypothesis that quality levels have equal mean alcohol content.

MAIN EFFECT: pH LEVEL  
F-statistic: 14.3968  
P-value: 0.014878  - compared to the alpha = 0.05, i.e. p_value < 0.05  

This evidence supports the alternate hypothesis that the pH level has a statistically significant effect on alcohol content, we reject the null hypothesis that pH levels have equal mean alcohol content.

### Residual Analysis:

We test the data for normality of the residuals, using the Shapiro-Wilk Normality Test, where the null hypothesis is that the data is normally distributed:  

Test Statistic: 0.975047 
P-value: 0.934117  
As the p_value > 0.05 - this evidence supports that the data is normally distributed  

(Placeholder for text describing the q-q plot)

![2w_anova_without_qqplot](2w_anova_without_qqplot.png)

(Placeholder for text describing the residuals vs Fittes Values plot)

![2w_anova_without_residuals_vs_fitted](2w_anova_without_residuals_vs_fitted.png)

## Conclusions:

In combination we can conclude that both wine quality and pH level have a statistically significant effect on alcohol content.  

Assessing the overall model, the R-squared = 0.9485, which can be interpreted as 94.85% of the variance in alcohol content is explained by the two factors (Quality and pH Level) combined.  The p-value of the overall model is 0.007698, which is less than the alpha = 0.05, therefore we can reject the hypothesis that the model isn't any good, i.e. we can conclude that the model is good.

We need next to assess the effect sizes of each factor, where the results of effect (η² - Eta-squared) are interpreted using Cohen's (1988) benchmarks for η²:

Quality effect size (η²): 0.5774 - i.e. 57.7% of the variance of alcohol content can be explained by the quality  
By Cohen's (1988) benchmarks for η², this constitutes a large effect.

pH Level effect size (η²): 0.3711 - i.e. 37.1% of the variance of alcohol content can be explained by the pH level  
By Cohen's (1988) benchmarks for η², this constitutes a large effect.

In summary the conclusions are that:
 - The model explains 94.85% of variance in alcohol content
 - Both factors should be considered when analysing wine characteristics
 - Results suggest that wine quality and pH chemistry relate to alcohol levels

It should be noted that there are important limitations with this test:
 - this is considered an observational study, and as such causation cannot be inferred  
 - sample represents single values (no replication within cells)
 - Results specific to the limited samples of wine

## Next steps:
Given the findings and limitations, and the limited number of observations, it is recommended to take additional measurements for each factor combination, and potentially increasing the analysis to include more factors.  Such data should be subjected to other analytical methods, such as 2-way ANOVA with replication.  This may highlight interaction effects between factors.

A formal test of the additivity assumption — Tukey's one-degree-of-freedom test for non-additivity — was not conducted in this analysis, with the assumption instead assessed visually via the interaction plot. Incorporating this test would strengthen the methodological rigour of the analysis: it directly tests whether any interaction between the two factors is present by introducing a single product term into the model, consuming exactly one degree of freedom from the residual. Given the small number of cells in a 3×3 no-replication design, where visual inspection of the interaction plot carries inherent limitations, this formal check is a meaningful extension.

## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/ANOVA_2-way_withoutRep_v2.py)

---

layout: default

title: Penguin Flipper Length by species and gender (2-Way ANOVA with Replication)

permalink: /2-way-anova-with-rep/

---

#### This project is in development

## Goals and objectives:

The business objective is understand the impact of species and gender on the flipper length of penguins, as well as the interaction of the two factors (species and gender).  The 2-Way ANOVA with replication technique was applied as there are multiple observations for each combination of species and gender.

The results provided evidence that both factors, as well as the interaction of both factors, have a statistically significant impact on the flipper length, and quantify the effect of each factor.

## Application:  

2-Way ANOVA with replication is a statistical method used to determine how two categorical independent variables (factors) affect a continuous dependent variable. The "with replication" part is crucial: it means testing multiple subjects or trials for every possible combination of the two factors.

This specific setup allows measurement of the Interaction Effect — whether the effect of one factor depends on the level of the other factor.

This is a powerful tool utilised by many sectors for multiple different reasons and benefits.

* **Manufacturing: Quality Control & Production** Engineers often use this tool to optimise machine settings across different shifts or materials.  
  * For example a car manufacturer tests the tensile strength of a metal part, where two primary factors exist; Machine Temperature (Low vs. High), and Supplier (Company X vs. Company Y).  This test can prevent "hidden" failures. This might find that Company X’s metal works perfectly at low heat, but fails miserably at high heat, whereas Company Y is stable across both. A standard one-way test would miss this interaction.  
* **Retail: Marketing and Consumer Behaviour** Retailers use this to understand how different demographics respond to various promotional strategies.   
  * For example a clothing brand measures Total Sales Volume, analysing the effect of two primary factors; Promotion Type (Discount Code vs. Free Gift), and Region (Urban vs. Rural).  This test can identify hyper-localization. The brand might discover that urban customers respond significantly better to "Free Gifts," while rural customers prefer "Discount Codes." This allows for targeted marketing spend rather than a "one size fits all" approach.  
* **Technology: Software & UX Performance** Technology companies use ANOVA to refine user experiences and optimise backend performance.  
  * For example a software company measures App Load Time, alongside two primary variables; Operating System (iOS vs. Android), and connection Type (4G vs. 5G).    This can determine if the interaction shows that iOS 4G users experience disproportionately high lag compared to all other groups, if so, the engineering team knows exactly where the bottleneck lies, rather than spending time optimising Android.  
* **Finance: Portfolio Management** Financial analysts use ANOVA to see how different market conditions impact various asset classes.  
  * For example an analyst tracks Quarterly Returns, with two features being; Economic Cycle (Recession vs. Growth), and Industry Sector (Tech vs. Healthcare).  This can support risk mitigation, through the identification of "defensive" sectors, as it is possible to statistically prove if a sector stays stable regardless of the economic cycle, while another sector's performance is heavily dependent on growth cycles. 

## Methodology:  

The 2-Way ANOVA with Replication technique was applied using the following method.

Initially the data was analysed to ensure that there is replication of observations for each combination of species and gender

Descriptive statistical analysis of the flipper length observations is undertaken, to better understand the data and detect any issues and outliers.  This includes analysis of the overall dataset, as well as analysis by species and gender.

The data is tested for normality for each combination of gender and species, using the Shapiro-Wilk test, as normality is an assumption of the 2-Way ANOVA with Replication test.

Another assumption to be tested is the homogeneity of the variances for each gender and species combination.  This was tested with Levene's Test.

The 2-Way ANOVA with Replication test was applied to the data determine the variance that can be explained by the model using the two factors.  This determines if each factor as well as the interaction of factors have a significant effect, as well as the size of the effect of each factor.

## Results and conclusions:

Results from the project related to the business objective.

### Descriptive Statistics:

Summary of the volume of observations for each combination of species and gender.

```
species    gender    count
Adelie     Female    73
           Male      73
Chinstrap  Female    34
           Male      34
Gentoo     Female    58
           Male      61
```
The overall distribution of the flipper lengths is plotted in the histogram below, including a KDE plot.  The Boxplot shows the distribution of flipper lengths by species and gender, which provides good insight into the data, and highlights that the species and gender are seemingly both factors in the flipper length.  The interaction plot of the mean flipper lengths simplifies the interactions, where the gradients reflect the difference in flipper length between male and female penguins for each species.  The 'steeper' the gradient, the larger the difference.

These charts suggest that for all species the mean flipper length for males is greater than for females, but the difference is not consistent across all species.  This will be further validated using the 2-Way ANOVA with replication tested.

![dist](2way_anova_with_dist.png)

![box](2way_anova_with_box_species_gender.png)

![interaction](2way_anova_with_interaction.png)

The summary descriptive statistics of flipper lengths by species and gender is:

```
                  count    mean   std    min    max
species   gender                                      
Adelie    Female     73  187.79  5.60  172.0  202.0
          Male       73  192.41  6.60  178.0  210.0
Chinstrap Female     34  191.74  5.75  178.0  202.0
          Male       34  199.91  5.98  187.0  212.0
Gentoo    Female     58  212.71  3.90  203.0  222.0
          Male       61  221.54  5.67  208.0  231.0
```

### Checking ANOVA assumptions:

An assumption of an ANOVA test is the normality of the values being analysed.  The histogram of the total set of data, as shown above, implies that overall the flipper length observations are not normally distributed, however the test for normality is to be undertaken for each combination of species and gender.

Using the Shapiro-Wilk test on each combination, the results are below.  The null hypothesis of the Shapiro-Wilk test is that the data is normally distributed, and as the p-value is greater than 0.05 for the test of each combination, we cannot reject the null hypothesis and the evidence suggests that the data is normally distributed as required.

```
Adelie, Female:		   p=0.4912 Normal
Adelie, Male: 		    p=0.4984 Normal
Chinstrap, Female: 	p=0.5074 Normal
Chinstrap, Male: 	  p=0.6201 Normal
Gentoo, Female: 	   p=0.2450 Normal
Gentoo, Male: 		    p=0.0545 Normal
```

Another assumption of an ANOVA test is of equal variances across the groups (i.e. the combinations of gender and species).

Levene's Test was applied, with the null hypothesis that the variances are equal.  Setting the confidence level equal to 0.05, the p-value of Levene's Test was calculated as 0.0365, which means that we reject the null hypothesis, and that there is evidence that the variances assumption may not be true.  As such it means the "spread" or dispersion of the data is not consistent across the different combinations of the two factors.

As the 2-Way ANOVA uses the F-statistic, which is a ratio of variances, when the underlying group variances are not equal, the F-test becomes less "robust".  This results in the 2-Way ANOVA potentially providing misleading results.  It is noted that the group sizes for each combination of factors, is not consistent, ranging from 34 to 73.  This can potentially lead to misleading ANOVA results.

Refering back to the boxplots, these highlight that there are multiple outliers associated to the 'Adelie' penguin species observations, which could potentially be causing the unequal variances.  These outliers are to be investigated further.

### 2-Way ANOVA Test application

The test produced an R² = 0.8396, i.e. ~84% of the variences in flipper length values can be explained by the two factors and the interaction of the 2 factors.

The p-value associated with each factor, including the interaction, determines if the each factor has a significant effect.  I.e. the null hypothesis is that the factor does not have an impact.  The p-value for each factor (species and gender) are of the order 10^-125 and 10^-24 respectively, and the p-value for the interaction is 0.0063, therefore we can say that there is evidence that each factor as well as the interaction of the factors are significant effects in the length of penguin flippers.

Given that there is evidence that the factors have an effect, the size of the effect of each was calcualted using the Eta-Squared values.  The table and chart below show the results of this which effectively state that the species accounts for ~77.4% of the variance, and the gender ~6% of the variance.  While the interaction effect is statistically significant, the size of the effect is 0.5%.  It is important to note that a factor being significant and the size of the effect are different factors, and just because the interaction effect is negligible, it does not mean that it is not statistically significant. Cohen's guidelines provide an interpretation of these sizes which are shown in the chart.  For completeness, the residuals represent ~16% of the variance, which can be interpeted as 16% of the variance is statistical randomness than the factors cannot explain - remembering that the R² of the model was 0.8396, so we had already seen that the model accounted for ~84% of the variance.

```
ADD
```

![effect](2way_anova_with_effect.png)


### Conclusions:

The conclusions are that the two factors (species and gender), as well as the interaction of the factors, have a satistically significant effect on the length of penguin flippers, however due to the unequal variances across groups that was reported, the results of this analysis are to be used with some caution.  

The species is the primary factor for flipper length, with the interaction of factors significant but with a negligable effect in size.

## Next steps:  

With any analysis it is important to assess how the model and application of the analytical methods can be used and evolved to support the business goals and business decisions and yield tangible benefits.

Recommendations and next steps for improving the analysis include:

* Improved observation data:
  * Collecting more observations and use this additional data to re-run the analysis
  * Outliers are to be investigated, for example those associated with the Adelie observations
* Further understand the interaction:
  * Consider the effect of gender for each species, and assess where the interaction changes across species.  For example the effect of gender on flipper size may be larger in one species than another.
  * Do pair-wise analysis of species, i.e. compare species A to species B, to understand 
* Addressing the unequal variances:
  * The ANOVA test can be robust even when variances are unequal, if the group sizes are equal or have minimal differences.  As such it is sugested to collect more data to enable the use of equal group sizes.
  * Understand the variances of each group in more detail to identify if which are the outliers and if there are any patterns to explain this.  In effect this is identifying the factor (species or gender) with the highest variance.
  * Investigate the use of a transformation on the flipper lenght measurements (e.g. log, or square root), to see if this stabilises the variances, and if so applying the 2-Way ANOVA to the transformed data.
  * Apply a version of ANOVA that does not assume equal variances, such as Welch's ANOVA, and interpret the results
* Expand the analysis:
  * Consider the expansion of the research to collect additional factors, or measurements, and undertake analysis of the effect of factors on other measurements (e.g. body mass, bill length etc.)
  * Expand the analysis to include more species of penguins
  * Perform analysis on blocks of data, for example in the penguin data a block could be a specific location, to determine if these are causing variance, and if there are additional factors that are impacting the measurements.

## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/ANOVA_2-way_withRep_penguins.py)

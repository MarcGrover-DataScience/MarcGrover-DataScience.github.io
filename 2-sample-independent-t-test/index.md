---

layout: default

title: Iris sepal petal lengths (Two-Sample Independent T-Test)

permalink: /2-sample-independent-t-test/

---

## Goals and objectives:

To test the sepal petal lengths of iris plants (species = enstata) grown under 2 separate conditions to determine if the lengths have a statistically significant difference, and hence if the growing conditions is a factor in the length of the sepal petal length. This project demonstrates how a Two-Sample Independent T-Test can be used to assess this null hypothesis using the data available.  The sample dataset includes 50 sepal petal length measurements from each of the 2 groups of ensata iris plants.   

## Application:  

The Two-Sample Independent T-Test is a powerful statistical tool used to determine if there is a statistically significant difference between the means of two completely separate (independent) groups. In a business context, this is essential for A/B testing, competitive analysis, and comparing performance between different segments or strategies.  

* An asset management firm tests two different portfolio construction strategies (A and B). They use the t-test to determine if the mean annual return of a sample of portfolios managed under Strategy A is significantly different from the mean annual return of a sample managed under Strategy B.
* Comparing the average credit score of applicants who default on loans versus those who do not default to validate a credit scoring model's predictive power.
* A retailer launches two different website layouts or email promotions (Version A and Version B) to two random, independent groups of customers. They use the t-test to compare the mean conversion rate or mean revenue per user between the two groups.
* A software company tests two different application workflows (Workflow 1 vs. Workflow 2) and measures the mean task completion time for two independent groups of new users to see which design is more efficient.
* After implementing a change to a production line process, a manufacturer compares the mean defect rate of products produced before the change to the mean defect rate of products produced after the change to quantify the improvement.

## Methodology:  

A workflow in Python was developed using libraries Scipy, Pandas and Numpy, utilising Matplotlib and Seaborn for visualisations.  The data came from a publically available dataset of iris measurements from the library scikit-learn, which was then extended to produce interesting statistical findings.  

The Two-Sample T-Test was used to test the null hypothesis that the mean sepal petal lengths are the same.  

The assumption of independence of observations is assumed due to design of the experiment.

Data preparation:  Minor transformation of data into a pandas dataframe for analytical purposes.

## Results and conclusions:

### Descriptive statistics:  

Initially a histogram and KDE of the iris petal lengths for each group of observations was created to visually inspect the distibution.    

![Histogram of petal length by group](/2s_ttest_hist.png) 

Boxplot and violin plots of the values for each group were also produced, to further understand the distributions. 

![Boxplot of petal length by group](/2s_ttest_boxplot.png) 
![Violin plot of petal length by group](/2s_ttest_violin.png) 

As initial visual inspection shows the means are different, but of we need to investigate further the statistical significance of the difference of means.

Simple descriptive statistics for each group:  
Ensata Group 1:  n=50, Mean=8.159cm, SD=0.381  
Ensata Group 2:  n=50, Mean=8.333cm, SD=0.401  
Difference in Means:  0.174cm 

It is noted that the histograms / KDEs look normal for each group, but we shall test that also.

### Hypothesis Test:

The Two-Sample T-Test was applied to the data as a whole, where the alpha was set to 0.05 - i.e. 95% confidence. The results were: 

T-Statistic: -2.3172  
P-Value: 0.0219  
As 0.0219 < 0.05 we can reject the null hypothesis (H₀) and conclude that the mean significantly differs from 6.0

Taking this further, we can further conclude that: 
95% Confidence Interval (CI) of the mean is: (5.710cm, 5.977cm) - noting 6.00mm is not within this range.

### Further investigation by species:

While analysing the data it was noted that there was a column stating the species of iris, so lets be inquisative and see if there are any patterns or insights from comparing the data for the species.

First we will produce the histograms (with associated KDE plots), and boxplots for each species:



Visually these plots suggest that the sepal petal length varies by species, so lets do some basic descriptive analysis by species, which shows that the mean sepal length by species, within one standard deviation is:  
Setosa: 5.006 ± 0.352 cm  
Versicolor: 5.936 ± 0.516 cm  
Virginica: 6.588 ± 0.636 cm  

From a simple assessment of these plots and results, it looks like the 'versicolor' species may have a sepal petal length mean of 6.0cm (the original hypothesised mean), so lets run a One-Sample T-Test for the data for that species, again with a 95% confidence interval set.  The results were: 

T-Statistic: -0.8767  
P-Value: 0.3849  
As 0.3849 > 0.05 we cannot reject the null hypothesis (H₀) and conclude that the supports the hypothesis that the mean is 6.0.

Taking this further, we can further conclude that: 
95% Confidence Interval (CI) of the mean of sepal petal length for the versicolor iris species is: (5.789cm , 6.083cm) - noting 6.00mm is within this range.

## Next steps:
Having concluded that the mean iris sepal length is not 6.0cm as hypothesised, there is evidence to support that one species 'versicolor' does have a mean sepal length of 6.0cm, whereas setosa and virginica species do not.  

It would be suggested that further analysis be undertaken to further test hypothesis that the mean sepal petal lengths are dirrent by species of iris, and determine the likely range for each species.  

Additional data would likley be gathered to support further analysis, and expand the species included, based on the business objectives.  2-way T-Tests and ANOVA methods could be used to gain further insight on each species and how they differ (or not) from each other.

## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/2Sample_Independent_T-Test_Pt2.py)

The original data used for the analysis is here:
[Access input data](/Iris_ensata.xlsx)

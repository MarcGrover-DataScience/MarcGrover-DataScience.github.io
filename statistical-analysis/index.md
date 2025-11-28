---

layout: default

title: Iris sepal petal lengths (One Sample T-Test)

permalink: /statistical-analysis/

---

## Goals and objectives:

Botanical researchers wish to undertake statistical analysis of iris sepal petal lengths, to test the hypothesis that the average length is 6.0cm.  This project demonstrates how a One-Sample T-Test was used to assess the null hypothesis, that the mean is 6.0cm, using the 150 observations of sepal petal length recorded.  

There was sufficient evidence to reject the null hypothesis, and supported the alternate hypothesis that the mean sepal petal length is not 6.0cm.  However based on the descriptive statistics, there was evidence that the mean sepal petal length was different for the three different species of iris in the sample.  Applying the One-Sample T-Test to the observations related to the iris species 'versicolor', supported the null hypothesis that the mean sepal length is 6.0cm for that species.

## Application:  

The One-Sample T-Test is a statistical tool used to determine if the mean of a single sample is significantly different from a known population mean or a hypothesised value. This is a core function in various business sectors for quality control, performance benchmarking, and validating assumptions.  

* An investment firm wants to know if the average annual return of a specific fund portfolio (the sample) is significantly different from a target benchmark like a market index (the hypothesized value)
* A bank needs to verify if its average loan approval time meets an internal service level agreement (SLA) target or a competitor's advertised time.
* A retailer sets a goal for the average customer satisfaction score (on a 1-5 scale) to be 4.5. They survey a sample of customers and use the test to see if the sample mean is statistically below that target.
* A tech company develops a new API and wants to ensure the average response time (latency) for a sample of API calls is less than a certain service performance standard (e.g., 50 milliseconds).
* A steel producer tests the tensile strength of a sample of a new alloy batch to determine if its mean strength differs from the specified standard for that grade of steel.

## Methodology:  

A workflow in Python was developed using packages Scipy, Pandas and Numpy, using Matplotlib and Seaborn for visualisations.  The data came from a publicly available dataset of iris measurements from the library scikit-learn.     

The One-Sample T-Test was used to test the null hypothesis that the mean sepal petal length is 6.0cm, which along with the visuals enabled full insight into the sepal petal lengths.  

Data preparation:  Minor transformation of data into a pandas dataframe for analytical purposes.

## Results and conclusions:

### Descriptive statistics:  

Initially a histogram and KDE of the iris petal lengths was created to visually inspect the distribution.    
![Histogram of petal length](/ttest_histogram.png)  

A boxplot of the values was also produced, to provide more understanding of the values, including the validation of absence of outliers.

![Boxplot of petal length](/ttest_boxplot.png)  

The mean sepal length is 5.843cm with a standard deviation of 0.828cm.  

An inspection of the images and descriptive analysis lead us to believe that the hypothesis may possibly be true, as 6.0cm is towards the centre of the distribution, but we can be more scientific that that.  

It is noted that the histogram looks broadly normal, but there appear to be three peaks of 'bins' which may require closer inspection.

### Hypothesis Test:

The One-Sample T-Test was applied to the data as a whole, where the alpha was set to 0.05 - i.e. 95% confidence level. The results were: 

T-Statistic: -2.3172  
P-Value: 0.0219  
As 0.0219 < 0.05 we can reject the null hypothesis (H₀) and there is evidence that the mean significantly differs from 6.0cm.

Taking this further, we can further conclude that: 
95% Confidence Interval (CI) of the mean is: (5.710cm, 5.977cm) - noting that 6.00cm is not within this range.

Please note that a test for normality was not undertaken in this example (this is covered in multiple other projects however), due to the Central Limit Theorem (CLT) and the size of population being above the common CLT threshold of 30.  Also, the histogram showed that the distribution was broadly normal with no significant outliers.  The T-Test is considered a robust test against deviations from a normal distribution.  Should the sample size be less than 30, then a test for normality would be more meaningful. 

### Further investigation by species:

While analysing the data it was noted that there was a column stating the species of iris, so lets analyse the observations by species so determine if there are any patterns or insights for each species.

The histograms (with associated KDE plots), and boxplots were generated for each species:

![Histogram of petal length by species](/ttest_histogram_species.png) 

![Boxplot of petal length](/ttest_boxplot_species.png)

Visually these plots suggest that the sepal petal length varies by species.  Basic descriptive analysis by species, showed that the mean sepal length by species, within one standard deviation is:  
Setosa: 5.006 ± 0.352 cm  
Versicolor: 5.936 ± 0.516 cm  
Virginica: 6.588 ± 0.636 cm  

From an assessment of these plots and results, it looks like the 'versicolor' species may have a sepal petal length mean of 6.0cm (the original hypothesised mean), so lets run a One-Sample T-Test for the observations of the 'versicolor' species, again with a 95% confidence interval set.  The results were: 

T-Statistic: -0.8767  
P-Value: 0.3849  
As 0.3849 > 0.05 we cannot reject the null hypothesis (H₀) and conclude that the data supports the hypothesis that the mean is 6.0cm.

Taking this further, we can further conclude that: 
95% Confidence Interval (CI) of the mean of sepal petal length for the versicolor iris species is: (5.789cm , 6.083cm) - noting 6.00cm is within this range.

## Next steps:
Having concluded that the mean iris sepal length is not 6.0cm as hypothesised, there is evidence to support that one species 'versicolor' does have a mean sepal length of 6.0cm, whereas setosa and virginica species do not.  

It would be suggested that further analysis be undertaken to further test hypothesis that the mean sepal petal lengths are different by species of iris, and determine the likely range for each species.  

Additional data would likely be gathered to support further analysis, and expand the species included, based on the business objectives.  2-Sample T-Tests and ANOVA methods could be used to gain further insight on each species and how they differ (or not) from each other.

## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/T-Test.py)

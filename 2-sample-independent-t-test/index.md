---

layout: default

title: Iris sepal petal lengths (2-Sample Independent T-Test)

permalink: /2-sample-independent-t-test/

---

## Goals and objectives:

To test the sepal petal lengths of iris plants grown under 2 separate conditions to determine if the lengths are the same. This project demonstrates how a 2-Sample Independent T-Test can be used to assess this null hypothesis using the data available.  This tests a sample of 50 sepal petal length measurements from each of the 2 groups grown in different conditions.  

## Application:  

The Two-Sample Independent T-Test is a statistical tool used to determine if the mean of a single sample is significantly different from a known population mean or a hypothesized value. This is a core function in various business sectors for quality control, performance benchmarking, and validating assumptions.  

* An investment firm wants to know if the average annual return of a specific fund portfolio (the sample) is significantly different from a target benchmark like a market index (the hypothesized value)
* A bank needs to verify if its average loan approval time meets an internal service level agreement (SLA) target or a competitor's advertised time.
* A retailer sets a goal for the average customer satisfaction score (on a 1-5 scale) to be 4.5. They survey a sample of customers and use the test to see if the sample mean is statistically below that target.
* A tech company develops a new API and wants to ensure the average response time (latency) for a sample of API calls is less than a certain service performance standard (e.g., 50 milliseconds).
* A steel producer tests the tensile strength of a sample of a new alloy batch to determine if its mean strength differs from the specified standard for that grade of steel.

## Methodology:  

A workflow in Python was developed using packages Scipy, Pandas and Numpy, using Matplotlib and Seaborn for visualisations.  The data came from a publically available dataset of iris measurements from the library scikit-learn.  

The one-sample T-Test was used to test the null hypothesis that the mean sepal petal lenght is 6.0cm.  

Data preparation:  Minor transformation of data into a pandas dataframe for analytical purposes.

## Results and conclusions:

### Descriptive statistics:  

Initially a histogram and KDE of the iris petal lengths was created to visually inspect the distibution.    


A boxplot of the values was also produced, to provide more understanding of the values.



Simple calcualtions mean 5.843cm with a standard deviation of 0.828cm.  

A casual inspection of the lead us to believe that the hypothesis may well be true, as 6.0cm is towards the center of the distribution, but we can be more scientific that that.  

It is noted that the histogram looks broadly normal, but there appear to be three peaks of 'bins' which may require closer inspection.

### Hypothesis Test:

The One-Sample T-Test was applied to the data as a whole, where the alpha was set to 0.05 - i.e. 95% confidence. The results were: 

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

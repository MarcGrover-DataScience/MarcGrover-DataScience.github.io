---

layout: default

title: Iris sepal petal lengths (One Sample T-Test)

permalink: /statistical-analysis/

---

## Goals and objectives:

To undertake statistical analysis of a set measurements of iris sepal petal lengths, to test the hypothesis that the average length is 6.0mm.  This project demonstrates how a T-Test can be used to assess this null hypothesis using the data available.  This tests the sample of 150 sepal petal length measurements against a hypothesised mean of the population.  

## Application:  

The One-Sample T-Test is a statistical tool used to determine if the mean of a single sample is significantly different from a known population mean or a hypothesized value. This is a core function in various business sectors for quality control, performance benchmarking, and validating assumptions.  

* An investment firm wants to know if the average annual return of a specific fund portfolio (the sample) is significantly different from a target benchmark like a market index (the hypothesized value)
* A bank needs to verify if its average loan approval time meets an internal service level agreement (SLA) target or a competitor's advertised time.
* A retailer sets a goal for the average customer satisfaction score (on a 1-5 scale) to be 4.5. They survey a sample of customers and use the test to see if the sample mean is statistically below that target.
* A tech company develops a new API and wants to ensure the average response time (latency) for a sample of API calls is less than a certain service performance standard (e.g., 50 milliseconds).
* A steel producer tests the tensile strength of a sample of a new alloy batch to determine if its mean strength differs from the specified standard for that grade of steel.

## Methodology:  

A workflow in Python was developed using packages Scipy, Pandas and Numpy, using Matplotlib and Seaborn for visualisations.  The data came from a publically available dataset of iris measurements from the library scikit-learn.  

The one-sample T-Test was used to test the null hypothesis that the mean sepal petal lenght is 6.0mm.  

Data preparation:  Minor transformation of data into a pandas dataframe for analytical purposes.

## Results and conclusions:

### Descriptive statistics:  

Initially a histogram and KDE of the iris petal lengths was created to visually inspect the distibution.    
![Histogram of petal length](/ttest_histogram.png)  

A boxplot of the values was also produced, to provide more understanding of the values.

![Boxplot of petal length](/ttest_boxplot.png)  

Simple calcualtions mean 5.843mm with a standard deviation of 0.828mm.  

A casual inspection of the lead us to believe that the hypothesis may well be true, as 6.0mm is towards the center of the distribution, but we can be more scientific that that.  
It is noted that the histogram looks broadly normal, but there appear to be three peaks of 'bins' which may require closer inspection.

### Hypothesis Test:

The T-Test was applied to the data as a whole and 

### Further investigation by species:

![Histogram of petal length by species](/ttest_histogram_species.png) 

![Boxplot of petal length](/ttest_boxplot_species.png)

## Next steps:


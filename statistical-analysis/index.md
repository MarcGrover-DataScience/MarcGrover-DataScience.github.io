---

layout: default

title: Test hypothesised mean length of Iris sepal petals (T-Test)

permalink: /statistical-analysis/

---

## Project Overview:

Statistical analysis of 150 sepal petal lenghts to test the hypothesis that the mean length is 6.0mm.    

### Goals and objectives

This project contains example statistical analysis of a set of data containing measurements of iris sepal petal lengths.  The hypothesis is that the average length is 6.0mm, and this shows how a T-Test can be used to assess this hythpothesis using the data available.  This tests a sample of the population against a hypothesised mean of the population.  

### Methodology:  

Python was used including packages Scipy, Pandas and Numpy.  The data came from a publically available dataset of iris measurements from the library scikit-learn.  
The one-sample T-Test was used to test the null hypothesis that the mean sepal petal lenght is 6.0mm.  

Data preparation:  Minor transformation of data into a pandas dataframe for analytical purposes.

### Results and conclusions

Initially a histogram and KDE of the iris petal lengths was created to visually inspect the distibution.  A casual inspection of this lead us to believe that the hypothesis may well not be true.  
![Histogram of petal length](/ttest_histogram.png)  

Descriptive analysis of the data showed that it has a mean 5.843mm with a standard deviation of 0.828mm.  

The T-Test was applied to the data as a whole and 

### Next steps


---

layout: default

title: Iris Sepal Lengths (One Sample T-Test)

permalink: /1-sample-t-test/

---

## Goals and objectives:

For this portfolio project, the business scenario concerns the iris flower dataset — a well-established reference dataset comprising 150 observations of sepal length, sepal width, petal length, and petal width across three species of iris: setosa, versicolor, and virginica. The objective is to determine whether the mean sepal length across the full dataset is statistically consistent with a hypothesised population value of 6.0cm, using a One-Sample T-Test as the inferential framework. 

The One-Sample T-Test is the appropriate technique here because the analysis involves a single continuous measurement being compared against a fixed reference value, rather than two groups being compared against each other. It answers the question: is the observed sample mean close enough to 6.0cm that the difference could plausibly be explained by sampling variation alone, or is there sufficient statistical evidence to conclude that the true population mean differs from that value? The test produces a t-statistic, a p-value, and a confidence interval for the mean, each of which contributes a different dimension of understanding to that question. 

A secondary objective emerges naturally from the structure of the dataset. The presence of three distinct species raises the question of whether the full-sample result is meaningful, or whether it masks material differences between species that a single aggregate test would obscure. This motivates a species-level investigation, examining whether the hypothesis of a 6.0cm mean holds when applied to each species individually — and in doing so, demonstrates how exploratory analysis can reveal structure in data that a single top-level test would miss entirely. 

By the end of the analysis, the project aims to demonstrate not only the correct application of the One-Sample T-Test, but also the analytical judgement to recognise when aggregate results warrant deeper investigation, and the ability to communicate statistical findings clearly to both technical and non-technical audiences.

## Application:  

The One-Sample T-Test is a statistical tool used to determine if the mean of a single sample is significantly different from a known population mean or a hypothesised value. This is a core function in various business sectors for quality control, performance benchmarking, and validating assumptions.  

* An investment firm wants to know if the average annual return of a specific fund portfolio (the sample) is significantly different from a target benchmark like a market index (the hypothesized value)
* A bank needs to verify if its average loan approval time meets an internal service level agreement (SLA) target or a competitor's advertised time.
* A retailer sets a goal for the average customer satisfaction score (on a 1-5 scale) to be 4.5. They survey a sample of customers and use the test to see if the sample mean is statistically below that target.
* A tech company develops a new API and wants to ensure the average response time (latency) for a sample of API calls is less than a certain service performance standard (e.g., 50 milliseconds).
* A steel producer tests the tensile strength of a sample of a new alloy batch to determine if its mean strength differs from the specified standard for that grade of steel.

## Methodology:  

The methodology adopted for this project follows the end-to-end data science workflow, progressing from data loading and exploratory analysis through to hypothesis testing and the extraction of business insight. The project is implemented in Python, using pandas for data manipulation, scipy for statistical testing, and seaborn and matplotlib for visualisation. Each stage of the pipeline is described below.

### Data Loading and Preparation
The dataset is sourced from scikit-learn's built-in load_iris() function. The raw array is converted into a pandas DataFrame for analytical purposes, with species labels added as a categorical column. The variable of interest for this analysis is sepal length, corresponding to the first column of the feature matrix. No further cleaning or transformation is required, as the dataset contains no missing values or erroneous records.

### Exploratory Data Analysis
Exploratory analysis is performed on the sepal length values across the full sample of 150 observations. Descriptive statistics are calculated including the mean, standard deviation, and standard error. Two charts are produced to support visual inspection of the data:

* A **histogram** with KDE overlay, used to assess the shape of the sepal length distribution and identify any obvious departures from normality or the presence of multiple sub-populations within the data.  
* A **boxplot**, used to inspect the central tendency, spread, and the presence or absence of extreme outliers that could distort the t-statistic.

The boxplot serves a dual purpose here, functioning both as an exploratory chart and as a visual check of the outlier assumption that underpins the One-Sample T-Test.  

### Hypothesis Testing — Full Sample
The null and alternative hypotheses for the full-sample test are:

* **H₀**: The mean sepal length of the iris population = 6.0cm
* **H₁**: The mean sepal length of the iris population ≠ 6.0cm

The One-Sample T-Test is performed using scipy.stats.ttest_1samp(), evaluated against a significance threshold of α = 0.05. A 95% confidence interval for the sample mean is also constructed, providing a plausible range for the true population mean and offering a complementary perspective to the p-value alone. Cohen's d is calculated as the effect size measure, defined as the difference between the sample mean and the hypothesised value divided by the sample standard deviation, quantifying the practical magnitude of any departure from H₀.

A formal normality test is not conducted for the full sample of 150 observations. This is justified by the Central Limit Theorem, which states that the sampling distribution of the mean will be approximately normal for sufficiently large samples regardless of the underlying population distribution. With n = 150 well above the conventional threshold of 30, and the histogram showing no severe skew or multimodality, the parametric t-test is appropriate without further distributional validation.  

### Species-Level Investigation
Visual inspection of the full-sample histogram reveals three distinct peaks in the distribution, suggesting the data may not be drawn from a single homogeneous population. This motivates a breakdown of the data by species, with descriptive statistics and distributional plots produced for each of the three groups separately. A forest plot of 95% confidence intervals is produced across all three species, with the hypothesised mean of 6.0cm shown as a reference line, providing a single consolidated visual that makes inter-species differences and their relationship to H₀ immediately apparent.

Where the species-level exploration identifies a group whose mean appears consistent with 6.0cm, a second One-Sample T-Test is applied to that subgroup, again at α = 0.05, with Cohen's d and a 95% confidence interval calculated alongside the test result.  

### Business Insight Extraction and Visualisation
The outputs of the hypothesis testing stages are synthesised into a structured interpretation that addresses the original business question directly. Results are communicated using clear, non-technical language supported by the visualisations produced during the exploratory and assumption-checking phases, ensuring the findings are accessible to both technical and non-technical stakeholders.

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

The Cohen's d was calculated as -0.189.

Please note that a test for normality was not undertaken in this example (this is covered in multiple other projects however), due to the Central Limit Theorem (CLT) and the size of population being above the common CLT threshold of 30.  Also, the histogram showed that the distribution was broadly normal with no significant outliers.  The T-Test is considered a robust test against deviations from a normal distribution.  Should the sample size be less than 30, then a test for normality would be more meaningful. 

### Further investigation by species:

While analysing the data it was noted that there was a column stating the species of iris, so lets analyse the observations by species so determine if there are any patterns or insights for each species.

The histograms (with associated KDE plots), and boxplots were generated for each species:

![Histogram of petal length by species](/ttest_histogram_species.png) 

![Boxplot of petal length](/ttest_boxplot_species.png)

The forest plot shows all three species' means and CIs against the μ=6.0 reference line, providing a powerful single visual that tells the whole story of the analysis at a glance.

![ttest_forest_plot_species](/ttest_forest_plot_species.png)

Visually these plots suggest that the sepal length varies by species.  Basic descriptive analysis by species, showed that the mean sepal length by species, within one standard deviation is:  
Setosa: 5.006 ± 0.352 cm  
Versicolor: 5.936 ± 0.516 cm  
Virginica: 6.588 ± 0.636 cm  

From an assessment of these plots and results, it looks like the 'versicolor' species may have a sepal length mean of 6.0cm (the original hypothesised mean), so lets run a One-Sample T-Test for the observations of the 'versicolor' species, again with a 95% confidence interval set.  The results were: 

T-Statistic: -0.8767  
P-Value: 0.3849  
As 0.3849 > 0.05 we cannot reject the null hypothesis (H₀) and conclude that the data supports the hypothesis that the mean is 6.0cm.

Taking this further, we can further conclude that: 
95% Confidence Interval (CI) of the mean of sepal length for the versicolor iris species is: (5.789cm , 6.083cm) - noting 6.00cm is within this range.

The Cohen's d was calculated as -0.124.

## Next steps:
Having concluded that the mean iris sepal length is not 6.0cm as hypothesised, there is evidence to support that one species 'versicolor' does have a mean sepal length of 6.0cm, whereas setosa and virginica species do not.  

It would be suggested that further analysis be undertaken to further test hypothesis that the mean sepal lengths are different by species of iris, and determine the likely range for each species.  

Additional data would likely be gathered to support further analysis, and expand the species included, based on the business objectives.  2-Sample T-Tests and ANOVA methods could be used to gain further insight on each species and how they differ (or not) from each other.

## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/T-Test.py)

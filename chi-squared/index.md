---

layout: default

title: Project (χ² Chi-Squared Test)

permalink: /chi-squared/

---

# This project is in development

## Goals and objectives:

The business objective is ...

## Application:  

In statistics, "Chi-squared" is an umbrella term for tests that use the χ² distribution to see if observed data matches what we’d expect by chance. 

The Chi-Squared (χ²) Goodness-of-Fit test is a non-parametric statistical hypothesis test used to determine how well an observed set of categorical data fits an expected distribution. Unlike the Chi-Squared Test of Independence (commonly used in A/B testing to compare two groups), the Goodness-of-Fit test compares a single sample against the distribution of a known population or a theoretical model.  The null hypothesis is that the data follows the expected distribution.

The test evaluates the "distance" between observed frequencies (O) and expected frequencies (E). If the calculated χ² value is significantly high, it indicates that the deviations between what was observed and what was expected are too large to be attributed to random chance, leading us to reject the null hypothesis that the data follows the specified distribution.

This approach is applicable across many industry sectors and scenarios. Practical examples showing where a Chi-Squared (χ²) Goodness-of-Fit test provides clear business value include:

🛍️ **Retail**: A clothing retailer uses the test to determine if the actual sales volume across different sizes (S, M, L, XL) matches the historical inventory distribution models.  

💻 **Technology**: A UX researcher applies the test to verify if the distribution of user clicks across five different navigation menu items is uniform, or if certain items are being disproportionately favored or ignored.  

🔬 **Science & Research**: A genetics researcher uses the test to confirm if the observed phenotypic ratios in a cross-breeding experiment align with the 9:3:3:1 ratio predicted by Mendelian inheritance laws.  

🏭 **Manufacturing**: A quality control engineer performs the test to check if the frequency of different types of defects (e.g., scratches, dents, or discolorations) matches the expected defect profile for a specific production line.

**Key Assumptions**:  To ensure the validity of the results, the following criteria must be met:
* **Categorical Data**: The variables must be nominal or ordinal.
* **Independence**: Each observation must be independent of the others.
* **Sample Size**: Each "cell" or category should have an expected frequency of at least 5.
* **Mutually Exclusive**: Each subject or item must fit into one, and only one, category.


Details of how this is applicable to multiple industries to solve business problems, generate insight and provide tangible business benefits. 


## Methodology:  

Details of the methodology applied in the project.

## Results:

Results from the project related to the business objective.

## Conclusions:

Conclusions from the project findings and results.

## Next steps:  

With any analysis it is important to assess how the model and application of the analytical methods can be used and evolved to support the business goals and business decisions and yield tangible benefits.


## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/t.py)

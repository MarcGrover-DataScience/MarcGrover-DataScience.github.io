---

layout: default

title: Project (χ² Chi-Squared Test)

permalink: /chi-squared/

---

# This project is in development

## Goals and objectives:

The business objective for this portfolio project is to apply a Chi-Squared Goodness-of-Fit test within a simulated quality control scenario at a fictional manufacturing facility. The facility operates a six-station production line, where defects arising during the manufacturing process should, under normal operating conditions, occur with equal probability across all six stations. Any significant deviation from this uniform distribution would indicate that one or more stations are underperforming — whether due to machine wear, calibration drift, operator variability, or a process fault — and would warrant targeted investigation. The analytical question is therefore whether the observed distribution of defects across the six stations is consistent with the expected uniform distribution, or whether the data provides statistically significant evidence of a systematic bias in where defects are occurring.  

The dataset is synthetically generated as part of the Python script, simulating 1,200 defect records assigned across six production stations. A subtle but deliberate bias is embedded in the data: one station produces defects at a meaningfully higher rate than the others, while the remaining stations are only marginally suppressed to compensate. This bias is intentionally designed to be non-obvious — inspection of the raw counts, proportions, and standard descriptive statistics returns values that appear broadly similar across all six stations, and no single chart makes the conclusion immediately clear. It is only through the application of the formal hypothesis test that the deviation from the expected uniform distribution can be identified with statistical confidence. This reflects a realistic and important class of analytical problem in quality control and operations management, where the signal is present in the data but subtle enough to evade detection without the correct statistical framework.  

The objective is to apply the Chi-Squared Goodness-of-Fit test, which assesses whether a single categorical variable follows a specified theoretical distribution — in this case, whether defect counts across the six stations follow the uniform distribution expected under a well-functioning process. Crucially, the test does not merely confirm whether differences exist between stations; it quantifies whether those differences are greater than would be expected from natural random variation alone. A key secondary objective is to decompose the test statistic to identify which specific station or stations are the primary drivers of any significant result, providing the kind of actionable, station-level diagnostic that a quality manager would need to prioritise an investigation.  

A further objective is to demonstrate correct application of the assumptions that underpin the Chi-Squared Goodness-of-Fit test, specifically the requirement that all expected cell frequencies are sufficiently large — conventionally, a minimum of five — to ensure the chi-squared approximation is valid. This assumption check is performed explicitly as part of the validation workflow, prior to any testing, reflecting the rigour required when applying inferential statistics in a professional analytical context.  

By grounding the analysis in a concrete operational scenario, this project demonstrates not only technical proficiency in Python-based statistical testing, but also the ability to translate a business problem into an appropriate statistical framework, validate the conditions under which that framework applies, and communicate findings in terms of practical significance — identifying not just that a problem exists, but where it is concentrated and what that means for the business.  

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

---

layout: default

title: Sleep Therapy Intervention (Paired Sample T-Test)

permalink: /paired-sample-t-test/

---

# This project is in development

## Goals and objectives:

The business objective is ...

## Application:  

A Paired Sample t‑Test (also called a dependent t‑test) is used to determine whether the mean difference between two related sets of observations is statistically significant. 

These “paired” observations come from the same subjects measured twice (before and after), or two conditions applied to the same participant, item, or process.

It answers the question:  “Did something change in a meaningful way?”

The technique computes the differences in observations, and using the mean and standard deviation of the differences, determines if this difference is statistically significant, and as a consequnce if there is a meaningful difference in the two sets of observations.

This approach is applicable across many sectors and scenarios.  Practical examples showing where a paired t‑test provides clear business value include:

🛍️ **Retail Sector**:
* Measure Impact of Store Layout Changes - Before‑and‑after comparison of customer dwell time, footfall flows, or basket size when a new layout is introduced.  This helps determine whether the redesign increases sales.
* A/B Testing In‑Store Promotions - Compare sales per customer before and after applying a new discount strategy in the same store.  This helps retailers optimise promotional return on investment.
* Training Effectiveness for Store Staff - Assess whether customer satisfaction scores for the same team improved after a training programme.

💻 **Technology Sector**:
* Software Performance Benchmarking - Compare system performance metrics (e.g., latency, CPU load) before and after code optimisation.  this identifies whether the new version truly improves performance.
* User Experience (UX) Improvements - Measure user completion time for tasks before and after a UI redesign.  This validates design choices based on statistically significant improvements.
* Cybersecurity Patch Impact - Compare false‑positive detection rates or scan times before vs after new threat‑detection algorithms.

🔬 **Science & Research Sector**:
* Clinical Trials & Experiments - Measure physiological indicators (e.g., heart rate, blood pressure) pre‑ and post‑treatment on the same subjects.
* Environmental Measurements - Assess changes in pollutant concentration before and after the introduction of a filtering system.
* Psychology & Behavioural Experiments - Compare participant scores on a cognitive task before and after an intervention such as mindfulness training.

🏭 **Manufacturing Sector**:
* Process Improvement (Lean / Six Sigma) - Compare defect rates from the same production line before vs after a process optimisation.
* Equipment Calibration Impact - Assess whether recalibration improves precision on the same machine.
* Energy Efficiency Testing - Compare power consumption of machinery before and after implementing efficiency controls.

A/B Testing and Paired Sample t‑Tests are related but significantly different.  A paired sample t‑test is a specific statistical test, whereas a A/B testing is an experimental framework that may use a t‑test (paired or unpaired), but also uses many other statistical methods.  For example, within this portfolio, there is an A/B Test example using chi-squared test of independence.

## Methodology:  

The methodology adopted for this project follows the end-to-end data science workflow, progressing from raw data through to the extraction and communication of business insight. The project is implemented in Python, using pandas for data manipulation, scipy for statistical validation, and seaborn and matplotlib for visualisation. Each stage of the pipeline is described in detail below.  The dataset used for the analysis is generated as part of the python script.  

**Data Loading**:  the dataset used for the analysis is generated as part of the python script.

**Exploratory Data Analysis**:

**Testing assumptions**:  A paired t‑test has three core assumptions, and each requires a specific diagnostic check.

* **Independence of Pairs (Between‑Pair Independence)** - Each pair of observations must come from independent subjects, i.e. one participant’s data must not influence another’s.  The matched pairs must consist of the same participants, but the pairs themselves must be independent from other pairs.  This is enforced by the generation of the dataset as part of the python script for this portfolio project.
* **Normality of the Difference Scores** - The differences (after – before) are tested for approximate normality, noting that the two sets of observations are not tested for normality. This is tested visually using a histogram with a KDE, a Q-Q plot and using the Shapiro-Wilk test for normality.  Note that should the test for normality be violated, then a non‑parametric alternative such as the Wilcoxon Signed‑Rank Test can be used.
* **No Extreme Outliers in the Difference Scores** - Outliers can distort the mean difference and inflate the t‑test statistic.  This is checked visualy using a boxplot of the difference scores.

**Business Insight Extraction and Visualisation**

## Results:

Results from the project related to the business objective.

## Conclusions:

Conclusions from the project findings and results.

## Next steps:  

With any analysis it is important to assess how the model and application of the analytical methods can be used and evolved to support the business goals and business decisions and yield tangible benefits.


## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/x.py)

---

layout: default

title: Teaching Methods Comparison (1-Way ANOVA)

permalink: /1-way-anova/

---

# This project is in development

## Goals and objectives:

For this portfolio project, the simulated business scenario concerns a fictional educational study involving 300 students assigned across three groups, each subject to a different teaching method, with the goal of determining whether the choice of teaching method produces a statistically meaningful difference in exam performance. A simple observation that the group means differ would be insufficient for this purpose — raw differences between sample means are always present to some degree, and without a formal statistical framework there is no basis for distinguishing a genuine effect of the teaching method from differences that could plausibly arise from natural variability between students.  

The objective is therefore to apply a One-Way Analysis of Variance (ANOVA), which tests whether the observed variation in exam scores between the three groups is greater than would be expected by chance alone. Critically, this project is designed to illustrate the value of statistical rigour in a scenario where the answer is not obvious from visual inspection alone. The three groups' score distributions overlap substantially, and no single chart makes the conclusion clear — it is only through the formal hypothesis test that a reliable conclusion can be reached. This reflects a realistic and important class of analytical problem: one where the signal exists in the data but requires the correct statistical machinery to extract it confidently.  

A key objective is to demonstrate correct application of the full ANOVA workflow, from assumption validation through to post-hoc testing. Where the ANOVA identifies that at least one group mean differs significantly from the others, it cannot on its own identify which specific pairs of groups are responsible for that difference. Tukey's Honest Significant Difference (HSD) post-hoc test is therefore applied to resolve this, controlling the family-wise error rate across all pairwise comparisons and identifying precisely where the significant differences lie. This two-stage structure — omnibus test followed by targeted post-hoc analysis — is central to the project and reflects the methodological precision required when comparing more than two groups.  

A secondary objective is to demonstrate awareness of the assumptions that underpin the One-Way ANOVA, specifically the requirements that observations are independent, that the residuals within each group are approximately normally distributed, and that the variances across groups are approximately equal (homoscedasticity). Each assumption is tested in practice — normality via the Shapiro-Wilk test and Q-Q plots, and homogeneity of variance via Levene's test. Where the equal variance assumption is questioned, Welch's ANOVA is additionally presented as the appropriate robust alternative, with both results compared to confirm the stability of the conclusion.  

By grounding every analytical decision in a clear methodological rationale, this project aims to demonstrate not only technical proficiency in Python-based statistical testing, but also the ability to select the right tool for the structure of the data, validate the conditions under which that tool is appropriate, and communicate findings in a way that is meaningful to both technical and non-technical audiences.

## Application:  

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
[View the Python Script](/DecisionTree_BreastCancer.py)

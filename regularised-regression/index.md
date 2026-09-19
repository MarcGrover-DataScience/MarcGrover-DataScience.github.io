---

layout: default

title: Project (Regularised Regression)

permalink: /regularised-regression/

---

# This project is in development

## Goals and objectives:

The business objective is to determine whether Ridge, Lasso, and Elastic Net regression produce more stable, generalisable house price predictions than an unpenalised Ordinary Least Squares (OLS) baseline when the feature set is large and features are correlated with one another — and, in doing so, to identify which property characteristics carry genuinely independent pricing signal versus which are redundant or noisy. The Ames Housing dataset (De Cock, 2011) — 2,930 residential property sales in Ames, Iowa, recorded across 82 fields — provides a realistic setting for this question: a much larger and higher-dimensional feature space than a typical regression dataset, with several groups of features (basement composition, above-ground living area) that are structurally related to one another by construction.

The analytical scope extends beyond a single model comparison. An OLS baseline is fitted first to establish a reference point, followed by Ridge, Lasso, and Elastic Net, each tuned via 10-fold cross-validated search over a shared grid of penalty strengths, ensuring differences in outcome reflect genuine differences between the penalty types rather than inconsistencies in the search process. Variance Inflation Factor (VIF) diagnostics are used to characterise the severity and nature of multicollinearity in the dataset before any model is fitted. Coefficient behaviour is then compared directly across all four models, with particular attention to two distinguishing properties: Lasso and Elastic Net's capacity to shrink coefficients to exactly zero — performing feature selection as a by-product of fitting — versus Ridge's shrinkage toward, but never to, zero. A regularisation path traces this behaviour continuously as the penalty strength increases.

This project is deliberately positioned differently from the [Multiple Linear Regression](https://marcgrover-datascience.github.io/multi-linear-regression/) project elsewhere in this portfolio. That project addressed OLS from an inferential standpoint — testing statistical significance, residual assumptions, and individual coefficient confidence intervals on a small, low-dimensional dataset. This project addresses a different failure mode of OLS entirely: coefficient instability and overfitting under a large, correlated predictor set — and a different remedy for it, penalisation rather than diagnosis-and-tolerate. Where that project found moderate multicollinearity (VIF ≈ 9.2) between two predictors and treated it as an acceptable, documented limitation, this project uses Ames' much more severe multicollinearity as the empirical basis for demonstrating why regularisation, rather than OLS with caveats, is the appropriate tool once a feature set reaches this scale.

The analysis confirms that regularisation delivers a material, measurable improvement under these conditions: Ridge, Lasso, and Elastic Net each outperform OLS on held-out test data (test R² of 0.91–0.91 versus 0.854 for OLS), and — more importantly for the stability question — the standard deviation of cross-validation R² across ten folds falls from 0.152 for OLS to 0.09–0.095 for the three regularised models. The VIF diagnostics further reveal that Ames' multicollinearity is not diffuse but concentrated in a small number of exact structural identities between features, an even more extreme condition than the elevated-but-finite VIFs the Multiple Linear Regression project encountered, and one that regularisation handles without incident where OLS coefficient estimates become arbitrarily unstable.





The business objective

## Application:  

Details of how this is applicable to multiple industries to solve business problems, generate insight and provide tangible business benefits. 


## Methodology:  

Details of the methodology applied in the project.

## Results:

Results from the project related to the business objective.

## Conclusions:

Conclusions from the project findings and results.

## Next steps:  

Next steps based on current results and conclusions from above and suggested follow-up actions, analysis etc.

## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/regularised_regression_ames_v2.py)

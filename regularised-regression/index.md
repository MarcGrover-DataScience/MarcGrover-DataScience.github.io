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

## Application:  

Regularised regression extends ordinary linear regression by adding a penalty term to the loss function that constrains the magnitude of the fitted coefficients. This single change addresses two problems that plague unpenalised regression as the feature set grows: overfitting, where a model fits training-set noise rather than genuine signal, and instability under multicollinearity, where correlated predictors cause coefficient estimates to swing wildly based on which observations happen to fall into the training sample. Ridge regression (L2 penalty) shrinks all coefficients toward zero without eliminating any, making it well-suited to situations where most predictors carry some genuine signal. Lasso (L1 penalty) can shrink coefficients to exactly zero, performing automatic feature selection alongside estimation — valuable when many candidate predictors are expected to be irrelevant or redundant. Elastic Net blends both penalties, offering a middle ground when the right degree of sparsity is not known in advance. Across all three variants, the practical benefit to a business is the same: more reliable, generalisable predictions from wide feature sets, without requiring an analyst to hand-select which of many candidate predictors to trust.

The business application of regularised regression spans any domain where predictive models are built from wide, and often correlated, feature sets:

🏠 **Real estate:**

**Hedonic property valuation**: as demonstrated in this project, automated valuation models draw on dozens of correlated property characteristics — size measures, quality ratings, age indicators — where regularisation prevents any single redundant feature from destabilising the price estimate.

**Portfolio-level risk modelling**: real estate investment trusts modelling rental yield or default risk across large, heterogeneous property portfolios use regularisation to keep coefficient estimates stable as new property types and features are added to the model over time.

🏦 **Finance:**

**Credit risk scoring**: consumer lending models often draw on hundreds of correlated bureau and behavioural features (utilisation ratios, payment history variables, account counts); Lasso-driven feature selection produces a leaner, more auditable scorecard without sacrificing predictive accuracy — a material advantage in a regulated setting where every retained feature must be justified.

**Factor model construction**: quantitative analysts regularise regressions of asset returns against large libraries of candidate risk factors, many of which are correlated with one another, to avoid attributing return to a factor that is merely a proxy for another.

🛍️ **Retail and marketing:**

**Marketing mix modelling**: media spend across channels (TV, paid search, social, display) is frequently highly correlated because campaigns run concurrently; Elastic Net is a standard choice in this setting, stabilising channel-level return-on-investment estimates that an unpenalised regression would otherwise attribute unreliably between correlated channels.

**Demand forecasting**: retailers forecasting product-level demand from wide feature sets — price, promotions, seasonality indicators, competitor activity — use regularisation to prevent overfitting when the number of candidate features is large relative to the number of historical observations for a given product line.

🧬 **Healthcare and life sciences:**

**Biomarker discovery**: genomic and clinical studies routinely measure far more candidate predictors (gene expression levels, biomarkers) than there are patients; Lasso's sparsity property is used directly as a feature-selection tool to identify the small subset of biomarkers most associated with an outcome, a setting where OLS is not merely unstable but frequently inapplicable outright.

**Clinical risk prediction**: hospital readmission or complication-risk models draw on large numbers of correlated clinical and administrative variables, where regularisation improves the reliability of risk scores used to prioritise post-discharge intervention.


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

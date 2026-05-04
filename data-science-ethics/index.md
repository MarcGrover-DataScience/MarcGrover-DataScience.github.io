---

layout: default

title: Ethics in Applied Data Science

permalink: /data-science-ethics/

---

# This project is in development


## Introduction — Why Ethics Matters in Applied Data Science

Machine learning and statistical modelling have moved rapidly from research settings into operational systems — informing credit decisions, clinical diagnoses, recruitment screening, fraud detection, and customer pricing, among many other domains. This shift changes the nature of the practitioner's responsibility. When a model's output influences a real decision with real consequences, technical performance alone is not a sufficient measure of quality. A model that achieves high accuracy on a held-out test set, but which cannot be explained to a stakeholder, produces systematically biased predictions for particular groups, or cannot be audited when a decision is challenged, is not a model that is ready for deployment.

Ethical consideration in data science is not a constraint imposed on top of technical work — it is part of what makes technical work credible and fit for purpose. The questions it raises are practical ones: Can the model's outputs be explained to the people they affect? Is the training data representative of the population the model will be applied to? Are the assumptions the model rests on visible and documented? What happens when predictions are wrong? These are not abstract concerns; they are the questions that practitioners, clients, and regulators increasingly ask before a model is trusted with consequential decisions.

This page does not attempt a comprehensive treatment of AI ethics — the literature on that subject is extensive and evolving. Its purpose is more specific: to set out the principles that inform the work in this portfolio, and to show how those principles connect to practical modelling decisions. Four themes are considered — **fairness**, **transparency**, **interpretability**, and **accountability & responsibility** — each examined both in general terms and with reference to the specific projects where they are most directly relevant.

The projects in this portfolio are built around realistic business scenarios across healthcare, retail, finance, and manufacturing. In each case, the ethical dimensions of the problem are as real as the technical ones. A false negative in a cancer diagnostic model carries a different cost from a false positive. A churn prediction model applied to customer retention raises questions about which customers are targeted and why. An anomaly detection system applied to transactions must be explainable to compliance teams and robust against the populations it monitors. Keeping these considerations in view — not as afterthoughts but as part of the problem framing — is what this page is intended to demonstrate.

## The Core Pillars of Responsible Data Science

Technical competence and ethical awareness are not separate concerns in applied data science — they are two dimensions of the same question: whether a model is genuinely fit for the purpose to which it will be put. A model may achieve excellent performance on a held-out test set and still be unsuitable for deployment if its outputs are unfair to particular groups, its reasoning is opaque to the people it affects, its assumptions are undocumented, or its predictions cannot be audited when challenged. Addressing these concerns is not a compliance exercise layered on top of the technical work — it is part of what makes technical work credible.

Four principles inform the approach taken across this portfolio: 

* fairness
* transparency
* accountability
* interpretability and explainability

These are not self-constructed criteria. They represent the most consistently cited themes across major AI ethics frameworks, including the EU High-Level Expert Group's _Ethics Guidelines for Trustworthy AI_ and the peer-reviewed literature synthesising published guidelines across the field. Each is considered in turn below — in general terms and with reference to the specific projects in this portfolio where it is most directly relevant.

### Fairness

Fairness is the most universally cited principle in AI ethics, and the most directly consequential. A model is unfair if it produces systematically different outcomes for different groups of people — whether by demographic characteristic, socioeconomic status, or any other attribute that should not influence the prediction — particularly when those outcomes carry real consequences for individuals.

Bias can enter the modelling pipeline at multiple stages. Training data may not be representative of the population the model will be applied to, causing the model to learn patterns that reflect historical inequity rather than genuine signal. Feature selection may inadvertently encode protected characteristics through correlated proxies. Evaluation metrics aggregated across the full dataset may conceal significant performance disparities between subgroups — a model that achieves 95% overall accuracy may perform substantially worse for a minority class or demographic segment that is underrepresented in the training data. Fairness-aware modelling requires that these questions are asked at the problem-framing stage, not as an afterthought once a model has been built.

Across this portfolio, fairness considerations are most directly relevant in two areas.  
* The **bank customer churn prediction project** applies logistic regression to a dataset that includes demographic features such as geography and gender. In this context, a deployed model must not target or disadvantage customers on the basis of characteristics unrelated to genuine churn risk — fairness requires that the model's retention recommendations are driven by behavioural signals rather than demographic proxies, and that performance is evaluated across customer subgroups, not only in aggregate. The class imbalance between churning and non-churning customers is also a fairness-adjacent concern: a model that optimises for overall accuracy at the cost of recall on the minority churn class fails precisely the customers the business most needs to identify.  This was addressed directly in the project through class weighting and evaluation using precision, recall, and F1-score alongside accuracy.
* The medical diagnostic projects — **Decision Trees**, **Random Forests**, **Gradient Boosted Trees**, and **SVM** — all applied to the Wisconsin Breast Cancer Diagnostic dataset — raise a related question: whether a model's false negative rate is acceptable in a clinical context where a missed malignant classification has direct consequences for patient outcomes. Fairness in this setting is not merely a question of demographic equity but of asymmetric error costs.

### Transparency

Transparency concerns the openness of a data science practitioner about what a model does, how it works, what data it was trained on, what assumptions underpin it, and where it may fail. A transparent model is one whose logic and limitations are visible and communicable — not necessarily to a technical specialist, but to the stakeholders whose decisions it will inform.

Transparency operates at several levels. At the data level, it means documenting the source, scope, and known limitations of the training data. At the methodological level, it means being explicit about modelling choices — why a particular algorithm was selected, what hyperparameters were applied, how the model was validated, and what performance benchmarks were achieved. At the output level, it means communicating predictions with appropriate context — including confidence, uncertainty, and the conditions under which the model's outputs should and should not be trusted.

The **[Data Science Workflow](/data-science-workflow/)** page sets out the structural commitment to transparency that underlies every project in this portfolio. Each project documents the full modelling process from problem framing and data acquisition through to evaluation, conclusions, and recommended next steps, with methodology described in sufficient detail to be reproducible. The business scenario framing adopted throughout the portfolio is itself a transparency mechanism — it makes explicit the intended deployment context of each model and the question it is designed to answer, rather than presenting a model as a generic technical exercise without deployment implications. The consistent documentation of performance metrics across projects — accuracy, precision, recall, F1-score, ROC-AUC — and the honest reporting of results that fall short of expectations (such as the Welch's ANOVA analysis in the One-Way ANOVA project, which returned a p-value of 0.061 and narrowly missed the threshold for significance) reflects the principle that transparency extends to negative and ambiguous findings, not only to confirmatory ones.


* Fairness — Bias, equitable outcomes, protected characteristics
* Transparency — being clear about what a model does, what data it uses, what assumptions it makes, and where it may fail. Link to Data Science Workflow page as an expression of this.
* Interpretability & Explainability — the distinction between globally interpretable models (Decision Trees) and post-hoc explainability tools (SHAP). Link directly to SHAP page and contrast with black-box models (GBT, Neural Networks, SVM).
* Accountability — who owns the output? What checks exist? Covers validation, documentation, and the importance of reproducibility. Link to Great Expectations and methodology documentation practices.

## Ethical Considerations by Model Type
A practical section mapping portfolio projects to their ethical risk profile — for example:

* High-stakes / sensitive domains: breast cancer diagnostic models (Decision Tree, Random Forest, GBT, SVM, MLP) — what are the consequences of false negatives vs. false positives?
* Fairness-sensitive: logistic regression on churn — does the model treat customer segments equitably?
* Opacity vs. interpretability: comparing simpler models (Decision Trees — fully interpretable) against ensemble/deep learning approaches, and how SHAP bridges that gap.

## Fairness & Bias Awareness
A concise section acknowledging that bias can enter at data collection, feature selection, and evaluation stages. No need to have run formal fairness audits — acknowledging the landscape and how I'd approach it (representative data, disaggregated metrics) is sufficient and honest.

## Regulatory & Industry Context
Brief coverage of the frameworks shaping this space — GDPR's right to explanation, the EU AI Act's risk-tiered approach, and sector-specific considerations (financial services, healthcare). Positions practitioner as someone aware of the environment practitioners operate in.

## Personal Commitment & Portfolio Reflection
A closing section tying it together — approach to building models that are not just accurate but explainable, documented, and fit for responsible deployment. This is a natural place to echo the framing from the homepage about "translating complex analysis into clear, actionable recommendations."


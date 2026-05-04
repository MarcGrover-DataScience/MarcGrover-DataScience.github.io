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


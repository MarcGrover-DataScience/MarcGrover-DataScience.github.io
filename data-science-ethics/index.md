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


---

layout: default

title: Ethics in Applied Data Science

permalink: /data-science-ethics/

---

## Overview

Machine learning models are increasingly applied to decisions that carry real consequences — in healthcare, finance, retail, and beyond. In these contexts, a model that is accurate but unfair, opaque, or unaccountable is not fit for deployment. This page sets out the ethical principles that inform the work in this portfolio and demonstrates how they connect to practical modelling decisions across the projects shown here.

Four principles are examined: **fairness**, **transparency**, **accountability**, and **interpretability & explainability**. These reflect the most consistently cited themes in major AI ethics frameworks, including the EU High-Level Expert Group's Ethics Guidelines for Trustworthy AI. The page also considers how ethical risk varies by model type and deployment context, how bias enters the modelling pipeline and how it is approached, and the regulatory landscape practitioners increasingly operate within.

This is not a comprehensive treatment of AI ethics — the literature on that subject is extensive and evolving. Its purpose is more specific: to show that responsible data science is not a constraint on the technical work in this portfolio, but part of what that work is.

## Introduction — Why Ethics Matters in Applied Data Science

Machine learning and statistical modelling have moved rapidly from research settings into operational systems — informing credit decisions, clinical diagnoses, recruitment screening, fraud detection, and customer pricing, among many other domains. This shift changes the nature of the practitioner's responsibility. When a model's output influences a real decision with real consequences, technical performance alone is not a sufficient measure of quality. A model that achieves high accuracy on a held-out test set, but which cannot be explained to a stakeholder, produces systematically biased predictions for particular groups, or cannot be audited when a decision is challenged, is not a model that is ready for deployment.

Ethical consideration in data science is not a constraint imposed on top of technical work — it is part of what makes technical work credible and fit for purpose. The questions it raises are practical ones: Can the model's outputs be explained to the people they affect? Is the training data representative of the population the model will be applied to? Are the assumptions the model rests on visible and documented? What happens when predictions are wrong? These are not abstract concerns; they are the questions that practitioners, clients, and regulators increasingly ask before a model is trusted with consequential decisions.

This page does not attempt a comprehensive treatment of AI ethics — the literature on that subject is extensive and evolving. Its purpose is more specific: to set out the principles that inform the work in this portfolio, and to show how those principles connect to practical modelling decisions. Four themes are considered — **fairness**, **transparency**, **interpretability**, and **accountability & responsibility** — each examined both in general terms and with reference to the specific projects where they are most directly relevant.

The projects in this portfolio are built around realistic business scenarios across healthcare, retail, finance, and manufacturing. In each case, the ethical dimensions of the problem are as real as the technical ones. A false negative in a cancer diagnostic model carries a different cost from a false positive. A churn prediction model applied to customer retention raises questions about which customers are targeted and why. An anomaly detection system applied to transactions must be explainable to compliance teams and robust against the populations it monitors. Keeping these considerations in view — not as afterthoughts but as part of the problem framing — is what this page is intended to demonstrate.

## The Core Pillars of Responsible Data Science

Technical competence and ethical awareness are not separate concerns in applied data science — they are two dimensions of the same question: whether a model is genuinely fit for the purpose to which it will be put. A model may achieve excellent performance on a held-out test set and still be unsuitable for deployment if its outputs are unfair to particular groups, its reasoning is opaque to the people it affects, its assumptions are undocumented, or its predictions cannot be audited when challenged. Addressing these concerns is not a compliance exercise layered on top of technical work — it is part of what makes technical work credible.

Four principles inform the approach taken across this portfolio: 

* Fairness
* Transparency
* Accountability
* Interpretability and Explainability

These are not self-constructed criteria — they represent the most consistently cited themes across major AI ethics frameworks, including the EU High-Level Expert Group's _Ethics Guidelines for Trustworthy AI_ and the peer-reviewed literature synthesising published guidelines across the field. Each is described below; specific portfolio applications are addressed in the sections that follow.

### Fairness

Fairness is the most universally cited principle in AI ethics, and the most directly consequential. A model is unfair if it produces systematically different outcomes for different groups of people — whether by demographic characteristic, socioeconomic status, or any other attribute that should not influence the prediction — particularly when those outcomes carry real consequences for individuals.

Bias can enter the modelling pipeline at multiple stages: training data may not represent the deployment population; feature selection may inadvertently encode protected characteristics through correlated proxies; and evaluation metrics aggregated across a full dataset may conceal significant performance disparities between subgroups. Fairness-aware modelling requires that these questions are asked at the problem-framing stage, not as an afterthought once a model has been built. How bias enters the pipeline at each stage, and how it is approached in this portfolio, is addressed in detail in the Fairness and Bias Awareness section below.

### Transparency

Transparency concerns the openness of a data science practitioner about what a model does, how it works, what data it was trained on, what assumptions underpin it, and where it may fail. A transparent model is one whose logic and limitations are visible and communicable — not necessarily to a technical specialist, but to the stakeholders whose decisions it will inform.

Transparency operates at several levels: at the data level, documenting the source, scope, and known limitations of the training data; at the methodological level, being explicit about modelling choices, validation procedures, and performance benchmarks; and at the output level, communicating predictions with appropriate context, including uncertainty and the conditions under which outputs should and should not be trusted. 

The **[Data Science Workflow](/data-science-workflow/)** page sets out the structural commitment to transparency that underlies every project in this portfolio — from problem framing through to conclusions and recommended next steps — with methodology described in sufficient detail to be reproducible. The honest reporting of ambiguous findings, such as the Welch's ANOVA result in the One-Way ANOVA project which narrowly missed the threshold for significance, reflects the principle that transparency extends to negative results as much as confirmatory ones.

### Accountability

Accountability addresses the question of ownership: who is responsible for a model's predictions, how can those predictions be audited when challenged, and what processes exist to detect and correct errors in deployed systems? It covers documentation, reproducibility, validation, and the practitioner's obligation to build models that can be interrogated after the fact.

In practice, this requires that data pipelines are documented, code is structured and commented, validation procedures are explicit, and the assumptions embedded in a model's design are recorded and retrievable. The Great Expectations project addresses accountability at the data pipeline level directly — implementing a formal validation framework that checks incoming data against defined expectations before it reaches the model, on the principle that accountability for a model's outputs begins with accountability for the data it consumes.

### Interpretability and Explainability

Interpretability and explainability concern the degree to which a model's predictions can be understood — either because the model's internal logic is directly readable, or because post-hoc tools can reconstruct a feature-level account of how a given prediction was produced. The distinction matters practically: an interpretable model is inherently transparent by design; an explainable model is opaque internally but can be accounted for after the fact using dedicated frameworks.

The most accurate models — ensemble methods, deep neural networks, SVMs with non-linear kernels — tend to be the least interpretable. Where predictions must be justified to stakeholders, clients, or regulators, post-hoc explainability frameworks bridge this gap. The SHAP project demonstrates this directly, applying SHapley Additive exPlanations to produce both global feature importance summaries and observation-level explanations of individual predictions. The interpretability risk profiles of different model types in this portfolio, and how that risk is managed, are addressed in the Ethical Considerations by Model Type section below.

## Ethical Considerations by Model Type

Not all models carry the same ethical risk. A forecasting model applied to retail inventory and a classification model applied to clinical diagnosis may be technically similar in construction but occupy entirely different positions in terms of the consequences of their errors, the populations they affect, and the standards of justification they must meet before being trusted. Understanding the ethical risk profile of a model — before it is built, not after — is part of responsible problem framing.

Three risk dimensions run through the projects in this portfolio: 

* High-stakes and Sensitive Domains
* Fairness-Sensitive Problems
* Opacity and Interpretability

These are not mutually exclusive categories. A single model can carry high-stakes risk, fairness-sensitive characteristics, and interpretability challenges simultaneously — the medical diagnostic models in this portfolio do exactly that. The purpose of considering them separately is to ensure that each dimension receives explicit attention rather than being absorbed into a generalised sense that a model is "complex" or "sensitive".

### High-Stakes and Sensitive Domains

A model operates in a high-stakes domain when the cost of a prediction error is materially asymmetric — when a false negative is not equivalent to a false positive, and when the consequences of being wrong extend beyond a misclassified row in a test set to a real decision with real effects on a real person or system. In these contexts, headline accuracy is a particularly inadequate summary of model quality. A model achieving 95% accuracy in a clinical diagnostic task may still produce a false negative rate that, at scale, represents a significant number of missed diagnoses. The relevant question is not only how often the model is right, but what happens when it is wrong, and whether the distribution of errors is acceptable given the deployment context.

High-stakes deployment also places additional demands on documentation, validation, and the communication of uncertainty. A model whose outputs will directly influence a clinical, financial, or safety-critical decision must be held to a higher standard of evidence than one informing a lower-consequence recommendation. Its performance must be evaluated on held-out data representative of the deployment population, its error types must be disaggregated rather than aggregated into a single metric, and its limitations must be communicated to the practitioners who will act on its outputs.

The clearest examples in this portfolio are the medical diagnostic models applied to the Wisconsin Breast Cancer Diagnostic dataset. The Decision Tree, Random Forest, Gradient Boosted Trees, Support Vector Machine, and Feedforward Neural Network (in development) projects all address binary classification of tumours as malignant or benign. In this context, the two error types are not equivalent: a false negative — classifying a malignant tumour as benign — may result in delayed treatment with direct consequences for patient outcomes, while a false positive — classifying a benign tumour as malignant — triggers unnecessary further investigation. Each project reports precision, recall, and F1-score alongside overall accuracy precisely because these metrics surface the distribution of errors across classes in a way that a single accuracy figure does not. The SHAP project extends this further, demonstrating that a model achieving 95.61% test accuracy on this task can additionally provide feature-level explanations of individual predictions — a capability that, in a genuine clinical deployment, would be a prerequisite for a clinician to act on the model's output with confidence. The high-stakes nature of the domain is what makes interpretability not a desirable addition to the model but a necessary one.

The Anomaly Detection (Isolation Forest) project (in development) raises a related set of high-stakes considerations in a different domain. Anomaly detection applied to financial transactions or operational systems operates in an environment where both error types carry significant cost: a missed anomaly may represent undetected fraud or equipment failure, while a false alert may trigger unnecessary intervention or erode confidence in the system. The asymmetry of consequences, and the question of who is accountable when an alert is missed or incorrectly raised, are defining characteristics of the deployment context, not peripheral concerns.

### Fairness-Sensitive Problems

A model is fairness-sensitive when its predictions have the potential to produce different outcomes for different groups of people — particularly where those differences track characteristics such as demographic background, geography, or other attributes that should not be legitimate determinants of the prediction. Fairness risk does not require discriminatory intent. It can arise from training data that over- or under-represents particular groups, from features that act as proxies for protected characteristics, or from optimisation objectives that maximise aggregate performance at the cost of equitable outcomes across subgroups.

Identifying a model as fairness-sensitive does not mean it cannot be built or deployed — it means that additional scrutiny is required. Performance should be evaluated disaggregated by relevant subgroups, not only in aggregate. The features driving predictions should be examined for the presence of demographic proxies. The choice of evaluation metric should reflect the cost of different error types across groups. The intended use of the model's output — who will act on it, and how — should be considered as part of the problem framing.

The bank customer churn prediction project is the most directly fairness-sensitive project in this portfolio. The dataset includes demographic features — geography and gender among them — alongside behavioural and account variables. In a deployed retention system, a model that assigns high churn probability disproportionately to customers sharing particular demographic characteristics — even without any explicit intent — risks producing recommendations that treat customers inequitably on grounds unrelated to genuine churn risk. The project addresses the class imbalance between churning and non-churning customers through class weighting and evaluation using precision, recall, and ROC-AUC rather than raw accuracy alone. This matters for fairness as well as performance: a model that optimises aggregate accuracy by systematically underperforming on the minority churn class fails precisely the customers the retention programme is designed to reach, and does so in a way that aggregate metrics conceal.

The Sentiment Analysis (Transformers) project (in development) raises a subtler fairness consideration. Transformer-based language models trained on large corpora inherit the biases present in that training data — associations between particular demographic groups and sentiment-laden language that may cause the model to assign systematically different sentiment scores to otherwise equivalent text depending on the identity of subjects or authors referenced. In a business context where sentiment analysis informs decisions about customers, employees, or public opinion, these inherited biases can propagate into recommendations at scale. The project applies a pre-trained model rather than training from scratch, which transfers this responsibility from the dataset curation stage to the model selection and evaluation stage: understanding the known limitations of the chosen model, and testing its outputs for systematic skew across relevant categories, is part of responsible deployment.

The K-Nearest Neighbours project, applied to wine quality classification using the UCI Red Wine Quality dataset, illustrates a fairness-adjacent concern that arises even in lower-stakes domains: class imbalance. The dataset's skewed distribution of quality ratings required rebanding of the target variable to produce a workable classification problem, addressing the risk that a model trained on the unmodified distribution would learn to ignore the tails of the quality spectrum — the rarest classes — in favour of dominant middle categories. While the consequences here are less acute than in clinical or financial contexts, the methodological response is the same: disaggregated evaluation and deliberate handling of distributional skew rather than acceptance of aggregate metrics that obscure minority-class performance.

### Interpretability Risk: Opacity in Models Requiring Explanation

A model carries interpretability risk when it is deployed in a context that requires its reasoning to be communicated — to a client, a regulator, a clinician, or an affected individual — and its internal logic is not directly accessible. This risk is not a fixed property of the model type alone. A Random Forest applied to a low-stakes internal optimisation problem may carry minimal interpretability risk; the same model applied to a credit decision or diagnostic classification carries significant interpretability risk because its outputs will be acted upon by people who need to understand and trust the reasoning, not only the headline prediction.

Interpretability risk increases in proportion to model complexity, the stakes of the application, and the degree to which affected parties have a legitimate interest in understanding how a decision was reached. It is most acute for ensemble methods and deep learning architectures, where the relationship between inputs and outputs is distributed across hundreds or thousands of parameters or trees and cannot be read directly. Managing this risk requires either selecting a simpler, inherently interpretable model where performance permits, or applying post-hoc explainability frameworks where a more complex model is necessary.

This tension is most explicitly explored across the supervised machine learning projects in this portfolio. The Decision Tree project applies the most interpretable model in the portfolio to the breast cancer diagnostic task — achieving 93.86% test accuracy with a classification logic that is fully readable as a sequence of binary splits on specific cell nucleus measurements. A clinician can follow the path from root to leaf for any individual patient and understand precisely which measurements triggered which classification at each step, without any additional tooling. The Random Forest that follows raises accuracy to 95.61% by aggregating 150 such trees, but that gain comes at the direct cost of this readability. No individual tree's path can be traced for a given prediction; the model's output is a probability aggregated across an ensemble whose internal logic is not directly accessible. The Gradient Boosted Trees model (97.37% accuracy) and the Support Vector Machine extend this further. Each step up the performance ladder increases the interpretability gap.

The SHAP project addresses this gap directly, applying SHapley Additive exPlanations to the Random Forest classifier to reconstruct feature-level explanations of individual predictions on the held-out test set. The analysis identifies five features that collectively account for 54.5% of the total mean absolute SHAP value — worst area, worst perimeter, worst concave points, mean concave points, and worst radius — and produces observation-level waterfall plots that decompose each individual classification into per-feature contributions against a baseline expectation. This is the mechanism by which a genuinely black-box model can be made accountable in a high-stakes deployment: not by simplifying the model, but by instrumenting it to explain itself.

The deep learning projects carry the most acute interpretability risk in the portfolio. The Feedforward Neural Network and LSTM Time Series models (in development) distribute their learned representations across multiple layers of weights and activations with no inherent correspondence to interpretable features. The Sentiment Analysis (Transformers) project (in development) applies a pre-trained transformer architecture whose scale and complexity make the relationship between input text and output sentiment score essentially opaque without dedicated explainability tooling. These models are appropriate where their performance advantage justifies their deployment and where the output is used as an input to human judgement rather than as an autonomous decision, but the interpretability risk they carry should be explicitly acknowledged and managed as part of any real deployment — not treated as an acceptable trade-off that goes unexamined.

## Fairness & Bias Awareness

Fairness in machine learning is not achieved by ignoring sensitive attributes or assuming that a model trained on historical data will naturally produce equitable outcomes. Bias is a structural risk that arises from the way data is collected, the way features are selected and constructed, and the way model performance is evaluated and reported. Understanding where it enters the pipeline — and what can be done to detect and mitigate it — is a prerequisite for deploying models responsibly.

This section does not present a comprehensive treatment of algorithmic fairness. Its purpose is more grounded: to describe the three principal stages at which bias risk is most acute, and to set out the approaches that inform the work in this portfolio.

### Bias at the Data Collection Stage

The most fundamental source of bias is the training data itself. A model learns patterns from its training data and applies them to new observations — so if that data does not accurately represent the deployment population, the model's learned patterns will reflect the training distribution, not the real world.

Underrepresentation is the most common form of this problem. Substantially fewer examples from particular demographic groups or population segments means weaker learned patterns for those groups — often without any signal in aggregate evaluation metrics. A model achieving 94% overall accuracy may be performing at 98% for the majority group and 78% for a minority group; the headline figure conceals the disparity entirely.

Historical bias is a related but distinct concern. Training data frequently reflects decisions made under conditions that were themselves inequitable — lending records, hiring decisions, or clinical referral patterns that encode systemic disparities. A model trained on this data learns to predict what the past would have decided, and encodes that as a forward-looking recommendation. Automating historical inequity through a well-specified model is not a technical failure — it is a failure of problem framing that begins with the data.

The practical response is to interrogate the dataset before modelling begins: examining the distribution of the target variable and key features across relevant subgroups, understanding how data was collected and whether collection methods produced systematic exclusions, and considering whether historical labels are themselves a reliable and equitable ground truth. Where underrepresentation is identified, approaches including oversampling, class weighting, and targeted data augmentation can partially address the imbalance — but they are mitigations, not corrections, and the limitations of the training data should be documented alongside the model's outputs.

### Bias at the Feature Selection Stage

Even with broadly representative training data, bias can enter through the features used to make predictions. Direct inclusion of protected characteristics — demographic group, gender, geographic location — as predictors is the most obvious risk. In practice, however, proxy features are the more insidious problem: variables that do not directly encode a protected characteristic but correlate with it strongly enough for the model to learn the association indirectly. Postcode may correlate with ethnicity; job title or educational institution may correlate with socioeconomic background. A model that excludes a sensitive attribute but retains correlated proxies will learn the association regardless.

The appropriate response is not necessarily to remove all features that correlate with protected characteristics — many such correlations carry genuine signal relevant to the prediction. The goal is to make the decision explicit: to identify which features carry this risk, assess whether their inclusion is justified by the prediction task, and evaluate whether outcomes are equitable given the features selected. Feature importance analysis and observation-level explainability tools such as SHAP provide a practical mechanism for this, making visible which features are driving predictions and whether any are disproportionately influential for particular subgroups.

### Bias at the Evaluation Stage

A model can be built on representative data with a carefully examined feature set and still produce inequitable outcomes if performance is evaluated using metrics that obscure disparities between groups. Aggregate metrics — overall accuracy, mean error, ROC-AUC across the full test set — report average performance and are insensitive to its distribution. They are necessary but not sufficient.

The most important correction is disaggregated evaluation: calculating performance metrics separately for relevant subgroups to identify whether the model performs materially differently across them. The choice of primary metric is itself a fairness-relevant decision. In imbalanced classification problems, overall accuracy rewards a model that performs well on the majority class while failing the minority. Recall on the positive class — the proportion of true positive cases correctly identified — is the metric that matters most when missing a positive case carries disproportionate cost, such as a malignant tumour classified as benign or a fraudulent transaction marked as legitimate.

This principle is applied consistently across the classification projects in this portfolio. The bank customer churn project evaluates both a baseline and feature-engineered logistic regression model using precision, recall, F1-score, and ROC-AUC alongside accuracy, with class weighting applied to address the imbalance between churning and non-churning customers. The K-Nearest Neighbours project required rebanding of the UCI Red Wine Quality target variable to correct a severely skewed class distribution that would have made headline accuracy a misleading guide to genuine model quality. The medical diagnostic projects — Decision Tree, Random Forest, Gradient Boosted Trees, SVM, and MLP — all report the full classification breakdown across both the malignant and benign classes, making the distribution of false negatives and false positives explicit rather than absorbed into an aggregate figure.

### Approaching Bias Awareness in Practice

Formal fairness auditing — comprehensive disaggregated analysis across demographic subgroups using dedicated toolkits — goes beyond the scope of projects built on publicly available datasets not specifically constructed for fairness research. What the portfolio does reflect is the underlying discipline: examining class distributions before modelling, selecting and justifying evaluation metrics rather than defaulting to accuracy, applying corrections for imbalance where identified, and using interpretability tools to make visible what the model has learned.

Bias awareness is not a checklist completed at the end of a project — it is an orientation that shapes how problems are framed, how data is interrogated, how models are evaluated, and how results are communicated. These questions do not always have clean answers, but the discipline of asking them is part of what responsible applied data science requires.

## Regulatory & Industry Context

The ethical principles discussed in this portfolio do not exist in isolation — they are increasingly reflected in formal regulatory frameworks and industry standards that govern how AI and machine learning systems are built and deployed. 

The EU Artificial Intelligence Act, which introduces a risk-tiered classification of AI systems with the most stringent requirements applied to high-risk applications in healthcare, finance, and employment, represents the most significant legislative development in this space. The EU High-Level Expert Group's Ethics Guidelines for Trustworthy AI provides an influential non-binding framework underpinning much of the academic and industry consensus on responsible AI practice. GDPR's right to explanation — the requirement that individuals subject to automated decisions have the right to a meaningful account of the reasoning involved — has direct implications for model interpretability in any consumer-facing deployment within its scope. Sector-specific standards, such as the Financial Conduct Authority's guidance on model risk management in financial services, extend these principles into domain-specific requirements that practitioners in those industries must navigate.

Awareness of this landscape is a practical requirement for any data scientist working on models intended for real deployment, not an academic interest. Understanding which regulatory category a proposed model falls into, and what obligations that carries, is part of responsible problem framing. The projects in this portfolio are built on publicly available datasets in a portfolio context rather than live operational systems, but they are deliberately framed around the sectors — healthcare, finance, retail, manufacturing — where these frameworks apply most directly. The ethical considerations built into the approach throughout this portfolio reflect the standards that responsible deployment in those sectors would require.

## Personal Commitment & Portfolio Reflection

The work in this portfolio is built around a consistent conviction: that a model which performs well on a test set but cannot be explained, has not been examined for fairness, or cannot be audited when its outputs are questioned, is not finished work. 

Accuracy is a necessary condition for a useful model, not a sufficient one. The commitment running through each project — to document methodology transparently, to evaluate performance using metrics that surface the full distribution of errors rather than flattering aggregates, to apply interpretability tools where the model's internal logic is not directly accessible, and to frame every analysis around a realistic deployment context — reflects this. These are not additions layered on top of the technical work; they are part of what the technical work is.

The ambition stated on the homepage of this portfolio — translating complex analysis into clear, actionable recommendations — carries an implicit ethical dimension that this page has tried to make explicit. A recommendation built on a model whose assumptions are undocumented, whose errors fall disproportionately on particular groups, or whose reasoning cannot be communicated to the person acting on it, is not a trustworthy recommendation regardless of the sophistication of the analysis behind it. Responsible data science and effective data science are not in tension. They are the same standard, applied consistently.

_Revision date: May 2026_

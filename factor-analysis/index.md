---

layout: default

title: Project (Factor Analysis)

permalink: /factor-analysis/

---

# This project is in development

## Goals and objectives:

For this portfolio project, the business scenario concerns the validation of a 50-item personality assessment used as part of a recruitment screening product. An HR-tech company has licensed the assessment — built on the International Personality Item Pool (IPIP-50) — on the understanding that it measures five distinct personality constructs: Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism, collectively known as the Big Five or OCEAN model. Before deploying the assessment at scale in hiring decisions, the business needs evidence that the instrument actually does what it claims: that its 50 individual survey items cleanly resolve into five coherent, distinct constructs, rather than blurring across traits in a way that would make the resulting scores unreliable. The analysis uses 19,718 anonymised responses to the IPIP-50 instrument, sourced from the Open-Source Psychometrics Project, with each item rated on a five-point scale and ten items per theorised trait.

This question — whether a claimed set of underlying constructs genuinely exists in the data — is precisely the question Factor Analysis is designed to answer, and it is worth being explicit about why this project sits deliberately apart from the [Principal Component Analysis](https://marcgrover-datascience.github.io/principal-component-analysis/) project elsewhere in this portfolio, despite both techniques operating on a correlated set of observed variables and producing a smaller set of derived dimensions. PCA is a variance-maximising technique: its components are linear combinations of the observed variables, constructed purely to capture the greatest possible variance, with no obligation — and no expectation — that any component corresponds to something real. The components are a mathematical by-product of compressing the data efficiently. Factor Analysis proceeds from the opposite premise. It is a latent-variable model that treats the five personality traits as real, unobserved dispositions that cause a person's answers to the 50 items, with each item also carrying variance specific to itself and unrelated to any shared trait. In this project, the factors are not a convenient summary — they are the entire object of investigation. The question is not "how few dimensions can this data be compressed into?" but "do the five constructs this instrument claims to measure actually exist?" A recruiter reading both project pages should come away understanding that PCA and Factor Analysis answer fundamentally different questions, even when applied to superficially similar data.

A primary objective of the project is to establish, before any factor is extracted, that the item correlation structure is actually suitable for Factor Analysis at all. Bartlett's Test of Sphericity and the Kaiser-Meyer-Olkin (KMO) measure of sampling adequacy are used as formal suitability diagnostics — a precondition-checking step with no direct equivalent in the PCA project, where no such test is required before proceeding. Principal Axis Factoring (PAF) is used as the primary extraction method, on the basis that it carries no distributional assumption about the underlying data and is therefore more defensible for ordinal, Likert-scale survey responses than Maximum Likelihood extraction, which assumes multivariate normality; Maximum Likelihood is nonetheless fitted alongside PAF as a compared alternative, to sense-check that the choice of extraction method is not materially altering the conclusions.

A second, closely related objective concerns how many factors the data itself supports. Rather than simply asserting five factors because five traits are theorised, the project applies standard retention heuristics — the Kaiser criterion and Horn's Parallel Analysis — and compares their output directly against the theorised five-factor solution. Where these heuristics disagree with theory, that disagreement is treated as an analytical finding in its own right rather than an inconvenience to be resolved by fiat, particularly given the well-documented tendency of eigenvalue-based retention rules to over-extract factors when applied to ordinal Likert data. The five-factor solution is then rotated using an oblique rotation (Promax), rather than the orthogonal rotation more commonly associated with PCA. This is a deliberate methodological choice, not a stylistic one: the Big Five traits are theoretically permitted to correlate with one another — Conscientiousness and Neuroticism, for example, are well documented as negatively correlated — and an oblique rotation is the only rotation family that allows the resulting factors to remain correlated, producing a genuine factor correlation matrix as a direct output of the analysis. This is a further point of contrast with PCA, where components are constructed to be uncorrelated by design, making a comparable correlation matrix uninformative even where it could be produced.

By the end of the analysis, the project aims to demonstrate the correct end-to-end implementation of Exploratory Factor Analysis — suitability diagnostics, extraction method comparison, empirical-versus-theorised factor retention, oblique rotation, and communality-based model fit assessment — while directly answering the business question the HR-tech scenario poses: whether the 50-item assessment can be trusted to measure five distinct, coherent personality constructs. Item-level loadings are interpreted against the theorised trait structure, any items that fail to load cleanly onto a single factor are flagged as a specific, actionable quality concern for the assessment's item bank, and factor scores are produced as the practical deliverable a recruitment scoring pipeline would consume in place of the 50 raw item responses.

## Application:  

Factor Analysis is a statistical technique used to identify a small number of unobserved, or latent, variables — known as factors — that explain the pattern of correlations observed among a larger set of measured variables. Where a dataset contains many variables that are themselves highly correlated with one another, Factor Analysis proceeds from the assumption that these correlations arise because the variables are all, to varying degrees, manifestations of a smaller number of underlying constructs that cannot be measured directly.

The core principle distinguishes Factor Analysis from other dimensionality reduction techniques such as Principal Component Analysis. While PCA constructs components that are simply linear combinations of the observed variables designed to capture maximum total variance, Factor Analysis explicitly models a latent variable structure: it assumes each observed variable is generated by a combination of a small number of common factors, shared across variables, plus variance unique to that individual variable. This makes Factor Analysis particularly suited to situations where the analyst has a substantive hypothesis that certain measured variables are indirect indicators of an underlying, unmeasurable construct — such as customer satisfaction, financial risk appetite, or employee engagement — rather than simply a tool for compressing data.

Once fitted, the resulting factor loadings — the strength of association between each observed variable and each underlying factor — allow the analyst to interpret and name the latent constructs, and to score each observation on these reduced, more meaningful dimensions rather than on the original, larger set of correlated measurements.

This approach is applicable across many sectors and scenarios. Practical examples showing where Factor Analysis provides clear business value include:

🛍️ **Marketing & Retail**:

**Customer satisfaction modelling**: A retailer analyses dozens of individual survey items and identifies that they load onto a small number of underlying factors, such as "service quality" and "value perception", simplifying reporting and prioritisation for improvement initiatives.

**Brand perception research**: Market researchers reduce a large battery of brand attribute ratings into a handful of interpretable underlying dimensions, such as "trustworthiness" and "innovation", that more efficiently summarise consumer perception.

**Customer segmentation inputs**: Marketing analysts derive a small set of latent behavioural factors from a wide range of purchasing and engagement variables, providing cleaner, less redundant inputs to downstream segmentation models.

👥 **Human Resources**:

**Employee engagement surveys**: HR teams reduce a lengthy engagement questionnaire into a small number of underlying factors — such as "management support" and "career development" — that are easier for leadership to act upon than dozens of individual survey items.

**Competency framework validation**: Organisational psychologists use Factor Analysis to validate that a set of performance assessment items genuinely measures the distinct underlying competencies they were designed to capture.

**Organisational culture measurement**: People analytics teams identify a small number of latent cultural dimensions underlying a broad set of workplace climate survey questions, supporting targeted culture initiatives.

🏦 **Finance & Economics**:

**Macroeconomic indicator summarisation**: Economists reduce a large panel of economic indicators — interest rates, inflation measures, employment figures — into a small number of latent factors representing broad economic conditions, used as inputs to forecasting models.

**Investment risk factor modelling**: Portfolio managers identify a small number of latent risk factors, such as market risk and credit risk, that explain the co-movement of returns across a large universe of assets, informing portfolio construction and risk management.

**Credit risk construct identification**: Lenders reduce a wide range of financial behaviour variables into a smaller number of interpretable underlying risk constructs, supporting more transparent credit policy design.

🔬 **Science & Social Research**:

**Psychometric test development**: Psychologists use Factor Analysis to confirm that a personality or aptitude questionnaire measures the distinct underlying traits it was designed to assess, a foundational step in validating any new measurement instrument.

**Public health survey analysis**: Researchers reduce large batteries of lifestyle and health behaviour questions into a smaller number of latent factors, such as "health consciousness", used as predictors in subsequent epidemiological models.

**Educational assessment design**: Educational researchers validate that a set of exam or assessment questions genuinely measures the distinct underlying academic skills they were intended to test, supporting fair and valid assessment design.


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
[View the Python Script](/t.py)

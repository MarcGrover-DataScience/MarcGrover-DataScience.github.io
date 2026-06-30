---

layout: default

title: Uncovering Hidden Data Quality Issues and Income Drivers in Census Data (Exploratory Data Analysis)

permalink: /exploratory-data-analysis/

---

## Goals and Objectives

Exploratory Data Analysis is the foundation on which every downstream modelling decision rests. Before a single predictive model is built, a structured EDA process must answer three questions: 

* is the data trustworthy?
* what does it actually contain?
* what patterns within it are likely to matter? 

Skipping this step does not save time — it defers cost, surfacing data quality problems and false assumptions much later in a project, when they are far more expensive to fix.

This project demonstrates a complete, structured EDA workflow applied to the UCI Adult Income dataset (also known as the "Census Income" dataset), a real-world extract of US census records used to predict whether an individual earns more or less than $50,000 per year. The dataset was deliberately chosen because it contains a genuine, non-obvious data quality issue — missing values disguised as a placeholder character rather than flagged as conventional nulls — making it a realistic test of whether an EDA process is rigorous enough to catch problems that a superficial pass would miss.

The specific objectives were to:

- **Validate data integrity** before any analysis takes place, including checks for duplication, disguised missingness, and implausible values.
- **Diagnose the pattern, not just the presence, of missing data**, and use that diagnosis to justify a specific, defensible treatment strategy rather than a default one.
- **Characterise the dataset's structure and key relationships** through univariate and bivariate visual analysis, directly tied to the business question the dataset exists to answer.
- **Compare manual, hypothesis-driven analysis against automated profiling**, to establish where each approach adds distinct value and where they corroborate one another.

## Application

EDA is not a preliminary formality; it is where the majority of a project's eventual reliability is determined. In a business setting, this dataset's structure mirrors a common scenario: an organisation holds a large administrative or survey-style dataset (HR records, loan applications, census extracts, customer onboarding forms) and wants to use it to predict an outcome — in this case, income band. Three findings from this project map directly onto recurring real-world problems:

- 📋 **Disguised missingness is a silent risk.** Missingness can arrive disguised in multiple, simultaneous forms within the same pipeline, and checking for only one form gives a false sense of completeness. This dataset encodes missing values as a literal `"?"` placeholder string in some records, while others are silently converted to genuine `NaN` by the data-loading library's own default behaviour before any custom validation logic runs. A standard `.isnull().sum()` check catches the second form but not the first, reporting only a fraction of the true missingness — a partial assurance that would still let most of a real data quality issue pass straight into modelling undetected. The same failure mode appears constantly in real systems: legacy exports mixing `"N/A"`, `"-1"`, `"0000"`, or blank strings alongside true nulls within the same column. Explicitly searching for every plausible placeholder form, rather than trusting a single null-detection method, is a habit that prevents a specific and common class of production data error.
- 🔍 **The pattern of missingness should drive the treatment, not a default rule.** Rather than reaching for mode imputation or row deletion by convention, this project diagnosed *why* values were missing before deciding *how* to treat them. The result changed the decision: missingness in `workclass` and `occupation` was found to co-occur in 99.6% of cases, consistent with individuals who are not currently employed in any conventional sense. Imputing those values with the dataset's most common category (`Private` workclass) would have manufactured a false signal; treating the absence as its own explicit, informative category preserved the truth in the data instead of overwriting it.
- ⚖️ **Automated profiling and manual analysis are complements, not substitutes.** An automated profiling report was generated alongside the manual analysis using `fg-data-profiling`. It independently corroborated the manual missingness finding, and surfaced two things the manual pass had not explicitly targeted: a near-perfect correlation between `education` and `education_num` (the same information encoded twice), and a second, smaller batch of duplicate rows that only became identical *after* missing values were standardised — a downstream consequence of the imputation decision itself. Running both approaches together caught more than either would have alone.

## Methodology

**Dataset**: UCI Adult Income dataset (Becker & Kohavi, 1996), retrieved via the `ucimlrepo` package (`fetch_ucirepo(id=2)`). The raw dataset comprises 48,842 records across 15 columns: 6 numeric features (`age`, `fnlwgt`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`) and 9 categorical features (`workclass`, `education`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country`, and the `income` target).

**Data validation** proceeded in stages:

1. **Exact duplicate detection.** 29 fully duplicated rows were identified and removed, reducing the working dataset to 48,813 records.
2. **Target standardisation.** The `income` target was stripped of whitespace and trailing full stops to resolve a known UCI export inconsistency (`">50K"` vs `">50K."`), producing two clean classes.
3. **Disguised missing value detection.** Every categorical column was explicitly checked for the literal placeholder string `"?"`, run alongside a standard `pandas` null check. Together, these surfaced two distinct disguised forms of missingness across three columns: values still encoded as the literal `"?"` string, and values already silently converted to genuine `NaN` by the data-loading library's own default CSV-parsing behaviour. Checking for either form alone would have caught only part of the true missingness.
4. **Pipeline defect detection and correction.** An initial implementation of the cleaning step applied a blanket string conversion to each categorical column before checking for placeholder values. This had an unintended side effect: any cell already holding a genuine `NaN` was silently converted into the literal string `"nan"`, which then bypassed imputation entirely and appeared as its own spurious category. The defect was caught by inspecting the resulting `workclass` distribution chart, traced to its root cause, and corrected; a safety-net validation check was then added to the script to automatically flag any stray string-typed placeholder value (`"nan"`, `"none"`, `"null"`, and case variants) that might bypass missing-value handling in future. This step underlines a broader principle: validating the analysis pipeline's own code is as important as validating the dataset itself.
5. **Missingness pattern analysis.** Before choosing a treatment, the co-occurrence of missingness across the three affected columns was quantified using a correlation of their missing-value indicators, to establish whether the gaps were structurally related or independent.
6. **Numeric range sanity checks.** Age values were checked against a plausible adult census range (16–100) to rule out data entry errors.

**Missing value treatment**: missing values in `workclass`, `occupation`, and `native_country` were imputed with an explicit `"Unknown"` category, rather than the column mode. This decision was made for three reasons, established during validation: it preserves the meaning of absence (a respondent who has never worked is a different reality to one whose employer was simply unrecorded), it avoids overwriting category proportions with imputed values that did not originate from that class, and it avoids manufacturing a false majority-class signal that the modelling stage would otherwise have to unlearn.

**Univariate and bivariate analysis** was conducted using eleven sequential, individually rendered Seaborn visualisations, covering the distribution of the target and key numeric/categorical features, and their bivariate relationship with income.

**Automated profiling**: a complementary profiling report was generated using `fg-data-profiling` (the actively maintained successor to `ydata-profiling`, renamed in April 2026), producing an independent statistical summary, correlation analysis, and automated data quality alert panel across all 15 variables.

## Results

**Income distribution.** Of the 48,813 valid records, 37,128 individuals (76.06%) earn $50,000 or less annually, and 11,685 (23.94%) earn more — a meaningfully imbalanced target that any downstream classification model would need to account for explicitly.

![Income class distribution](plot_01_income_class_distribution.png)

**Disguised missingness.** Missingness in this dataset took two disguised forms: a literal `"?"` placeholder string in some records, and genuine `NaN` values in others, introduced silently by the data-loading library's own default CSV-parsing behaviour before any custom validation logic ran. Across the three affected fields, this amounted to 2,799 missing values in `workclass` (5.73%), 2,809 in `occupation` (5.76%), and 856 in `native_country` (1.75%) — 6,464 affected cells in total, 0.88% of the dataset. Checking for the placeholder string alone would have caught only part of the true missingness — confirming that validation must check for every form a dataset's pipeline can produce, not just the one a single tool happens to flag.

**Missingness co-occurrence.** `workclass` and `occupation` missingness were almost perfectly correlated (0.998), with 2,799 of 2,809 occupation-missing records (99.6%) also missing `workclass`. This pattern is consistent with individuals outside conventional employment, and directly justified imputing both with an explicit `"Unknown"` category rather than the column mode.

![Workclass distribution including imputed Unknown category](plot_05_workclass_distribution.png)

**Age and working hours.** The median respondent age was 37 (IQR 28–48), and median working hours were 40 per week, with a long right tail extending to the dataset's maximum of 99 hours.

![Age distribution](plot_02_age_distribution.png)

**Relationship with income.** Higher-income individuals (`>50K`) were visibly older and worked more hours per week on average than lower-income individuals, evident in both boxplot comparisons.

![Age distribution by income band](plot_06_boxplot_age_by_income.png)

Educational attainment and marital status showed a clear gradient against the proportion earning more than $50,000, with advanced degree holders and married individuals showing substantially higher proportions in the higher income band than the dataset average.

![Proportion earning more than 50K by education level](plot_08_income_proportion_by_education.png)

**Correlation structure.** Pairwise linear correlation among the six numeric features was weak throughout, with the strongest relationship — `education_num` and `hours_per_week` — reaching only 0.144. This is not a data quality concern; it reflects a genuine property of the dataset. Numeric correlation cannot capture the dataset's strongest income-related signal, since marital status, occupation, and education category — all categorical — carry far more discriminative information than the six continuous columns do on their own.

![Correlation heatmap of numeric features](plot_11_correlation_heatmap.png)

**Automated profiling corroboration.** The fg-data-profiling report, regenerated against the corrected dataset, independently flagged 9 alerts. The redundancy between education and education_num is flagged, as is a comparable correlation between relationship and sex. Notably, the workclass/occupation correlation flagged in the earlier (uncorrected) run no longer appears: with missingness in both columns now correctly unified under a single "Unknown" category rather than split across two inconsistent labels, the tool's correlation threshold is no longer triggered — though the manual co-occurrence analysis above (99.6%) confirms the underlying relationship is, if anything, unchanged. This is a useful illustration of a broader point: an automated tool's binary alert threshold and a manual, continuous-valued metric can disagree even when describing the same relationship, which is exactly why both approaches are run side by side rather than relying on either alone. The tool also flagged 23 duplicate rows in the cleaned dataset and corroborated the imbalance in race (65.7%) and native_country (82.7%), and zero-inflation in capital_gain (91.7%) and capital_loss (95.3%).

![Automated profiling alerts panel](eda_alerts.png)

![Automated profiling overview panel](eda_overview.png)

## Conclusions

This project demonstrates that a structured EDA process delivers value distinct from, and prior to, any predictive modelling step. Four conclusions follow directly from the results:

- **Trust in a dataset cannot be established from a single missing-value convention.** Part of this dataset's missingness was already converted to standard `NaN` by the data-loading library itself, while the rest remained disguised as a literal placeholder string — checking for only one of the two forms would have caught some, but not all, of the true gaps. Real data quality assurance requires actively searching for every way a dataset's source system is known to encode absence, not just trusting whichever single check is run first.
- **Validating the pipeline matters as much as validating the data.** An early implementation of the cleaning logic introduced its own defect, silently converting genuine `NaN` values into the literal string `"nan"` and allowing them to bypass imputation. This was caught by inspecting the resulting chart output rather than assumed correct, and a safety-net check was added to guard against the same defect recurring. A rigorous EDA process treats the analysis code itself as something to be validated, not only the dataset it operates on.
- **The right treatment for missing data depends on why it is missing, not a default convention.** Diagnosing the 99.6% co-occurrence between `workclass` and `occupation` missingness was the step that justified treating both as an informative `"Unknown"` category, rather than applying mode imputation by default. This decision preserved a genuine signal in the data that a less careful pipeline would have erased.
- **Manual and automated EDA approaches are most powerful in combination.** The automated profiling report did not replace the manual analysis — it corroborated two independent findings, and surfaced a third (the education/education_num redundancy, and the post-imputation duplicate rows) that the manual pass had not directly targeted. Relying on either approach alone would have produced an incomplete picture.

Beyond the dataset itself, the underlying workflow — validate before trusting, diagnose before treating, and corroborate manual judgement with automated tooling — applies directly to any project beginning with an unfamiliar or third-party dataset.

## Next Steps

This EDA establishes a validated, well-understood foundation for two natural follow-on projects already mapped out on the portfolio roadmap:

- **Predictive classification.** The clear income-band imbalance (76.06% / 23.94%) and the bivariate patterns identified here — particularly the strength of categorical features over numeric ones — set up a natural classification project on this same dataset, with class imbalance handling as an explicit, pre-justified requirement rather than an afterthought.
- **The `education` / `education_num` redundancy** flagged by the automated profiling report should be resolved before any modelling stage, by retaining only one of the two encodings, to avoid artificially inflating feature importance or introducing multicollinearity.
- **Capital gain and capital loss zero-inflation** (91.7% and 95.3% zeros respectively) would benefit from being modelled as a two-part feature — a binary "has capital activity" flag alongside the continuous amount — rather than as raw continuous variables, an approach well suited to a future feature engineering or model interpretability project.
- **The `fg-data-profiling` Alerts panel** also flagged high imbalance in `race` and `native_country`; this is a useful early input to any future examination to determine if model performance is consistent across demographic subgroups.  This is a key consideration regarding AI ethics anf fairness.

## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/EDA_v0.2.py)

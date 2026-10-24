---

layout: default

title: Health Insurance Charges (Bootstrap Confidence Intervals)

permalink: /bootstrap-confidence-intervals/

---

# This project is in development

## Goals and objectives:

**The business problem.** A regional health insurer is preparing to launch a new product. Before committing, the pricing team has gathered a small pilot sample of 100 policyholders and their annual charges. The head of underwriting must set three numbers from it:

* **The typical annual charge (median).** This anchors the base premium.
* **The ratio of mean charges for smokers to non-smokers.** This drives the premium loading applied to smokers.
* **The 90th-percentile charge.** This informs the threshold at which stop-loss or reinsurance cover should begin.

A pilot of 100 cannot pin any of these down exactly. The useful question is not "what is the number?" but "how wrong could it plausibly be?", because a premium loading built on an estimate of 4.0 means something very different if the plausible range is 3.9–4.1 than if it is 3.3–4.9.

**The analytical question.** How precisely can these three quantities be estimated from a pilot of 100 policyholders, and can the method used to express that uncertainty be trusted when charges are heavily skewed?

**How the answer would be used.** The interval around the smoker ratio tells underwriting how much margin to build into the loading. The interval around the 90th percentile indicates how firmly a reinsurance threshold can be set, or whether it should wait for more data. Comparing interval widths at pilot sizes of 50, 100 and 200 answers a practical question directly: is the pilot big enough, or is more recruitment worth the cost?

**Why not a standard test?** Charges are right-skewed, and two of the three quantities have no tidy textbook interval. The median and the 90th percentile have no simple standard-error formula, and the ratio of two means has only an approximate one that leans on normality. The bootstrap sidesteps this by estimating the uncertainty directly from the sample itself. It is also a fair question whether the bootstrap can be relied upon, so the project is structured in two parts:

* **Part A (applied):** bootstrap intervals for the three quantities from a single pilot sample, checked against known population values.
* **Part B (validation):** a simulation study of how often bootstrap intervals contain the truth, across four distributions, five sample sizes and four interval methods, including a case where the bootstrap is known to fail.

### Objectives

1. Build and compare bootstrap intervals (percentile, basic, BCa and, for the mean, studentised) against a textbook t-interval.
2. Quantify how interval width shrinks as the pilot grows from 50 to 200 policyholders.
3. Measure empirical coverage of each interval method in a controlled simulation, and identify where each breaks down.
4. Draw practical guidance on which interval method to use, and when not to use the bootstrap at all.

The value of the project lies in the method and the honest account of its limits, not in any pricing figure: the data is simulated, and nothing here is a real premium recommendation.

## Application:  

**The technique.** The bootstrap estimates the uncertainty of a statistic by treating the sample as a stand-in for the population. It repeatedly resamples the observed data with replacement, recomputes the statistic each time, and reads the spread of those recomputed values as a measure of how much the statistic would vary from sample to sample. Different ways of turning that spread into a confidence interval give the percentile, basic, BCa and studentised methods compared in this project.

**When it is used.** It earns its place when:

the statistic has no convenient formula, such as a median, percentile, ratio or a model's performance metric;
the data is skewed or the sample is too small to trust normal approximations;
or an analyst simply needs a method that works the same way for any statistic, so that one tool covers many questions.

Its main benefits are generality, minimal distributional assumptions, and the fact that it can be implemented in a few lines and run cheaply on a standard laptop. Its main limitations are that it is only as good as the sample, can under-cover badly in small or heavily skewed samples, and fails outright for some statistics, such as the sample maximum. Part B examines each of these.

This approach is applicable across many sectors and scenarios. Practical examples showing where this provides clear business value include:

🏦 **Insurance and finance**  
Actuaries and risk teams need uncertainty ranges for tail quantities such as value-at-risk and large-loss thresholds, and for loss ratios between segments. These are exactly the median-, ratio- and percentile-type statistics studied here, usually estimated from limited claims history.  

🏥 **Healthcare and clinical research**  
Hospital costs and lengths of stay are strongly skewed, so a t-interval on the mean cost can be unreliable in small trials. Bootstrap intervals for mean or median cost differences between treatments support health-economic evaluations of whether a new treatment is worth its price.

🚆 **Operations and logistics**  
Service-level agreements are typically phrased as percentiles ("90% of deliveries within X hours"). A bootstrap interval around a delivery-time percentile shows whether an observed result is comfortably inside the target or merely lucky.

🛍️ **Retail and machine learning**  
Revenue per customer is skewed, so experiment teams often bootstrap differences in medians or trimmed means rather than relying on normal approximations. The same resampling idea underlies bagging in the Random Forest project, and bootstrapping the held-out test set gives an interval around metrics such as accuracy, where the point estimate alone (for example, 95.61% on 114 observations) conveys little about how much to trust it.  

**Relationship to existing portfolio pages.** The One-Sample T-Test page relies on the Central Limit Theorem at n = 150, and the A/B Testing page uses normal-approximation intervals for conversion rates. The bootstrap is the natural alternative when those assumptions are doubtful: skewed continuous outcomes, small samples, or statistics beyond the mean.

Details of how this is applicable to multiple industries to solve business problems, generate insight and provide tangible business benefits. 

## Methodology:  

### Dataset

The analysis uses the **Medical Cost Personal Dataset** (insurance.csv), available at https://www.kaggle.com/datasets/mirichoi0218/insurance. It holds 1,338 rows and seven columns: age, sex, BMI, number of children, smoker status, US region and charges. There are no missing values.

**What charges represents, and what it doesn't.** The dataset's own documentation describes charges only as "individual medical costs billed by health insurance". That wording is ambiguous: it does not say whether the figure is a claim payout, an amount billed, or something closer to a premium, and the data carries no claim dates, deductibles or co-payments. The data offers some clues. Charges rise sharply with smoking, age and BMI, as medical costs do. Every one of the 1,337 records is positive, with a minimum of $1,121.87 and no zeros, which real claims data would not look like, since many policyholders claim nothing in a given year. This project therefore treats charges as annual medical cost per policyholder, and makes no claim that these are payouts.

**The data is simulated.** The dataset is described as synthetic, generated from US Census Bureau demographic statistics for use in a machine-learning textbook, not drawn from real insurance records. Nothing here is evidence about real premiums or real people. The business scenario is a framing device for demonstrating the technique, and the write-up treats it as such.

### Known features of the data:

* **One exact duplicate row**, identical in all seven columns including charges to four decimal places, which makes coincidence very unlikely. It was removed, leaving 1,337 rows. One row in 1,338 changes nothing material, but the decision is recorded rather than hidden.  
* **Skewed charges**: skewness 1.51 overall, with a mean of $13,279 well above the median of $9,386.  
* **A mixture, not a single skewed shape**. Skewness comes almost entirely from the 1,063 non-smokers (1.54), while the 274 smokers (20.5% of the data) are close to symmetric (0.13) and cost far more on average. Chart 01 shows the combined distribution and chart 02 splits it by smoker status.

### Analytical design

The "known population" device. A real pilot never reveals the true answer, so it cannot show whether an interval was right. To get around this, the full cleaned dataset is treated as a known population, with true values for the mean ($13,279.12), median ($9,386.16), smoker-to-non-smoker ratio of mean charges (3.80) and 90th percentile ($34,832.74). This is a finite dataset, not a true infinite population, so the values are "truths" for the exercise only.

The pilot. One simple random sample of 100 policyholders was drawn without replacement, with a fixed seed. It was not stratified, because a real pilot could not guarantee how many smokers it would recruit. It contained 22 smokers.

Interval methods. For each statistic, 95% intervals were built by:

the percentile method (the middle 95% of the bootstrap distribution);
the basic (reverse-percentile) method;
BCa (bias-corrected and accelerated), which adjusts for bias and skewness;
the studentised (bootstrap-t) method, for the mean only, since it needs a standard error for every resample and a nested bootstrap for the other statistics was not worth the cost;
a standard t-interval for the mean as a textbook baseline.
Key methodological decisions
Stratified resampling for the ratio. Smokers and non-smokers were resampled separately, which holds group sizes fixed. Plain resampling would sometimes draw very few or zero smokers, adding noise the real pilot did not have.
B = 9,999 resamples for the pilot, 999 for the simulation. With B + 1 = 10,000, the 2.5% and 97.5% cut-points fall exactly on whole order statistics, and tail percentiles need many resamples to be stable. The simulation builds hundreds of thousands of intervals, so it uses 999 (with the same property) to stay within CPU limits.
Independent random streams. Each stage draws from its own seeded stream, so changing one stage cannot silently alter another's results.
Validation of the implementation. The mean and median intervals were checked against scipy.stats.bootstrap. Endpoints agree to within 0.1–2.6% of interval width, which is the size of difference expected from Monte Carlo noise alone.
Reporting asymmetry. Each interval's lower and upper arms are reported separately, since the t-interval's symmetry is exactly the assumption the bootstrap relaxes.
Is the pilot big enough? (pilot-size sweep)

To answer the practical question, 200 independent pilots were drawn at each of n = 50, 100 and 200. Each statistic's percentile interval (B = 999) was recorded as a width relative to the population value, with the median across the 200 pilots reported so one lucky or unlucky draw cannot dominate. Because samples come from only 1,337 rows, the bootstrap ignores a finite-population effect that matters slightly at n = 200 (15% of the data), which makes those intervals a little too wide.

Part B: coverage simulation

A single pilot cannot show whether a method is reliable, so Part B repeats the exercise on distributions where the truth is known exactly. Data was generated from four settings: Normal(0, 1), Lognormal(0, 1), Lognormal(0, 1.5) and Uniform(0, 1), with the statistics being the mean and median for the first three and the sample maximum for the Uniform. Each setting was run at n = 10, 20, 50, 100 and 200, with 1,000 replications per setting and 999 resamples each.

Empirical coverage is the share of replications in which the interval contained the truth, compared with the nominal 95%. With 1,000 replications, the Monte Carlo standard error is about 0.7 percentage points, so differences smaller than roughly ±1.4 points are indistinguishable from chance. The simulation also records which side misses fall on, as a method can be acceptable overall yet badly one-sided.

The lognormal settings are deliberately much more extreme than the insurance data. Their theoretical skewness is about 6.2 and 33.5, against 1.5 for the charges, so they act as a stress test rather than a mirror of Part A. The Lognormal(1.5) setting was added in a second version of the script, placed last in the run order so the original three settings reproduce the first version's results exactly.

Transparency and ethics

In line with the commitments on the Ethics in Applied Data Science page, the project states its limits plainly:

the data is simulated and its meaning is partly ambiguous, as set out above;
a single pilot's result, however tidy, is an anecdote, not evidence of coverage;
smoking status is a sensitive rating factor in pricing, and the ratio computed here illustrates the technique, not a recommended premium loading. Real use of such a factor would be shaped by regulation and fairness considerations beyond this project.

All code is seeded and reproducible, and the full scripts are shown in the Code section.

## Results:

Results from the project related to the business objective.

## Conclusions:

Conclusions from the project findings and results.

## Next steps:  

Next steps based on current results and conclusions from above and suggested follow-up actions, analysis etc.

## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/t.py)

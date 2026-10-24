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

Relationship to existing portfolio pages. The One-Sample T-Test page relies on the Central Limit Theorem at n = 150, and the A/B Testing page uses normal-approximation intervals for conversion rates. The bootstrap is the natural alternative when those assumptions are doubtful: skewed continuous outcomes, small samples, or statistics beyond the mean.

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
[View the Python Script](/t.py)

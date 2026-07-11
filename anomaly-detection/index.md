---

layout: default

title: Project (Anomaly Detection (Isolation Forest))

permalink: /anomaly-detection/

---

# This project is in development

## Goals and objectives:

The business objective is to demonstrate how an unsupervised anomaly detection model can support fault monitoring in an industrial manufacturing setting, where machine failures are rare, costly, and not reliably predictable using fixed rules or thresholds on individual sensors.

In many real-world manufacturing environments, historical examples of failure are too scarce, too varied in cause, or too poorly labelled to train a reliable supervised classifier. This project simulates that constraint directly: Isolation Forest is trained without access to failure labels, learning to isolate anomalous combinations of sensor readings — air and process temperature, rotational speed, torque, and tool wear — purely from the structure of the data itself. The dataset's true failure labels are withheld from training and used only afterwards, to evaluate how well the unsupervised model's flags align with genuine equipment failures.

A central theme of this project is that the two ways the model can be wrong are not equally costly. A missed anomaly may allow an undetected fault to progress towards unplanned equipment failure and production downtime; a false alert may trigger an unnecessary inspection or intervention, consuming maintenance resource and, if frequent, eroding operator trust in the monitoring system. Rather than treating the model's sensitivity as an arbitrary setting, this project treats the contamination parameter as a business decision, tuned against a stated cost asymmetry between these two outcomes.

This asymmetry also raises a question that extends beyond the model itself: who is accountable when a monitoring system misses a genuine fault, or when it raises an alert that turns out to be unwarranted? This question is explored further in the Ethics in Applied Data Science page, which references this project as a case study in high-stakes automated decision-making.

## Application:  

Isolation Forest is an unsupervised machine learning algorithm designed specifically for anomaly detection — the task of identifying rare observations that differ substantially from the majority of the data, without requiring any labelled examples of what an anomaly looks like. This makes it particularly valuable in real-world settings where anomalies are, by definition, rare and their exact form is often unknown in advance.

The core principle behind Isolation Forest inverts the logic used by most other anomaly detection methods. Rather than first building a profile of "normal" data and then measuring how far new points deviate from it, Isolation Forest works by attempting to isolate each observation through a series of random feature splits, building an ensemble of random decision trees. Because anomalies are, by definition, few in number and different in their feature values from the rest of the data, they tend to be isolated by very few random splits — resulting in a short average path length from the tree's root. Normal observations, densely surrounded by similar points, require substantially more splits to isolate, resulting in longer average path lengths. An anomaly score is derived directly from this path length across the ensemble of trees, with shorter average paths indicating a higher likelihood of anomaly.

This approach offers two significant practical advantages over distance- or density-based anomaly detection methods. First, it scales efficiently to large, high-dimensional datasets because it never needs to compute pairwise distances between observations. Second, because it directly targets the isolation of anomalies rather than modelling the distribution of normal data, it performs well even when the definition of "normal" is complex or the data does not conform to a well-behaved statistical distribution.

This approach is applicable across many sectors and scenarios. Practical examples showing where Isolation Forest provides clear business value include:

🏦 **Finance**:

**Fraud detection**: Payment processors flag transactions with unusual combinations of amount, location, and timing for review, without requiring pre-labelled examples of every possible fraud pattern.

**Anti-money laundering monitoring**: Compliance teams identify accounts exhibiting transaction patterns that deviate sharply from typical customer behaviour, supporting the detection of previously unseen laundering techniques.

**Market surveillance**: Trading venues detect unusual order patterns that may indicate market manipulation, flagging activity that departs from established trading norms for further investigation.

🏭 **Manufacturing**:

**Predictive maintenance**: Factories monitor sensor readings from industrial equipment in real time, flagging vibration, temperature, or pressure patterns that isolate quickly from normal operating behaviour as early indicators of impending failure.

**Quality control**: Automated inspection systems identify manufactured units with unusual combinations of dimensional or material properties, catching defect types that were never explicitly defined in advance.

**Process monitoring**: Process engineers detect abnormal combinations of process parameters on a continuous production line, enabling intervention before an anomaly develops into a costly batch failure.

💻 **Cybersecurity & Technology**:

**Network intrusion detection**: Security teams identify network traffic exhibiting unusual patterns of volume, protocol, or destination, flagging potential intrusions that do not match any previously catalogued attack signature.

**Account takeover detection**: Platforms flag user sessions with login or activity patterns that deviate sharply from an account's established behaviour, supporting early detection of compromised accounts.

**System health monitoring**: DevOps teams detect unusual combinations of infrastructure metrics — CPU, memory, latency — that may indicate an emerging system fault before it triggers a full outage.

🏥 **Healthcare**:

**Patient monitoring**: Intensive care systems flag combinations of vital sign readings that deviate sharply from a patient's own baseline, supporting earlier clinical intervention.

**Insurance claims review**: Health insurers identify claims with unusual combinations of procedure, cost, and provider characteristics for further review, supporting fraud and billing-error detection.

**Medical device quality assurance**: Manufacturers monitor device telemetry data for anomalous readings that may indicate a device malfunction, supporting proactive recall or maintenance decisions.


## Methodology:  

Details of the methodology applied in the project.

1 Dataset and Business Context  
2 Exploratory Data Analysis  
3 Preprocessing and Feature Preparation  
4 Model Selection Rationale (why Isolation Forest / why unsupervised)  
5 Contamination Tuning Against a Cost Function  
6 Model Training  

## Results:

**Exploratory Data Analysis**

![01_feature_distributions](01_feature_distributions.png)

_Figure 1: Distributions of the five numeric process and sensor readings._ Air and process temperature are both mildly multimodal, rotational speed is right-skewed, torque is approximately normal, and tool wear is close to uniform across its range — consistent with how the AI4I dataset was synthetically generated.

![02_correlation_heatmap](02_correlation_heatmap.png)

_Figure 2: Correlation matrix of the numeric features._ Air and process temperature are strongly correlated (0.88), as expected since process temperature is generated as a small offset above air temperature. More significantly, rotational speed and torque are strongly negatively correlated (−0.88), consistent with the machine holding power roughly constant under normal operation — this relationship directly motivated the power engineered feature described below.

![03_feature_boxplots_by_failure](03_feature_boxplots_by_failure.png)

_Figure 3: Boxplots of each numeric feature, split by true failure status._ Torque and rotational speed show a visible median shift under failure (higher torque, lower speed), but all five features overlap substantially between failed and non-failed observations. No single feature offers a clean separating threshold, motivating a multivariate, isolation-based approach over simple rule-based monitoring.

![04_failure_mode_counts](04_failure_mode_counts.png)

_Figure 4: Frequency of each of the five individual failure modes._ Counts range from 115 (HDF) down to 19 (RNF); failure modes are not mutually exclusive, so a single failed observation can trigger more than one flag. The low counts for several modes, particularly RNF, limit how much confidence can be placed in per-mode detection rates calculated later in this section.

**Iterative Feature Engineering**

Three versions of the feature set were evaluated, each built on evidence from the previous version's results rather than speculative tuning:

```
Version     Feature set                           Recall   Precision  F1
v1          Raw sensor readings only              14.2%    19.2%      0.163
v2          + power, temp_diff, overstrain        26.3%    19.8%      0.226
v3 (final)  + overstrain_ratio (Type-normalised)  58.1%    14.6%      0.233
```

The move from v1 to v2 added three features reflecting the physical mechanisms behind AI4I's failure modes — power (torque × angular velocity), temp_diff (process minus air temperature), and overstrain (tool wear × torque) — on the reasoning that Isolation Forest can only isolate an interaction between sensors if that interaction is presented to it directly, rather than left for random splits to rediscover. This raised recall from 14.2% to 26.3% while precision held broadly steady, confirming a genuine improvement rather than a shift along the same precision–recall trade-off.

The move from v2 to v3 normalised the overstrain feature by its true, product-Type-specific threshold (11,000 / 12,000 / 13,000 minNm for Types L / M / H respectively, per AI4I's documented generating logic), since overstrain failure is only meaningful relative to that Type-dependent threshold. This nearly doubled OSF detection (see per-mode breakdown below) and, combined with the contamination sweep re-optimising to a higher value once the model could better isolate genuine outliers, lifted overall recall to 58.1% — at a corresponding cost to precision, discussed further below.

**Contamination Tuning Against Cost Ratios**

![05_cost_curve_contamination_sweep](05_cost_curve_contamination_sweep.png)

_Figure 5: Weighted cost (ratio × missed failures + false alerts) across a swept range of contamination values, for three illustrative cost ratios._ The optimal contamination rises sharply with the assumed cost ratio: 0.030 at 5:1, 0.135 at 10:1, and 0.180 at 20:1. This sixfold shift in operating point, driven by only a fourfold change in assumed relative cost, is the clearest evidence in this project that contamination functions as an encoding of business judgement rather than a purely technical setting — a small change in how costly a missed failure is assumed to be produces a large change in how the model actually behaves in deployment.

The 10:1 ratio was carried forward as the primary evaluation point, reflecting an assumption that a missed failure is meaningfully, but not extremely, more costly than an unnecessary inspection. This is stated as an illustrative assumption rather than a costed figure, consistent with the qualitative cost framing agreed for this project.

**Model Evaluation**

![06_anomaly_score_distribution](06_anomaly_score_distribution.png)

_Figure 6: Distribution of Isolation Forest decision function scores (lower = more anomalous), split by true failure status._ The distribution is genuinely bimodal rather than a single blurred boundary: a distinct left tail, composed almost entirely of true failures, represents cases the model isolates with high confidence. A substantial share of failures nonetheless sits within the main hump shared with normal observations — these are the failures the model is structurally less able to distinguish, discussed further below.

![07_confusion_matrix](07_confusion_matrix.png)

_Figure 7: Confusion matrix of predicted anomaly against actual failure, at the chosen 10:1 operating point._ Of 339 genuine failures, 197 were flagged (true positives) and 142 were missed (false negatives); of 9,661 normal observations, 1,153 were flagged unnecessarily. This corresponds to a recall of 58.1% and precision of 14.6% (F1 = 0.233) — in practical terms, the model catches close to three in five genuine failures, at a cost of fewer than one in six flagged observations actually being a failure.

![08_failure_mode_detection_rates](08_failure_mode_detection_rates.png)

_Figure 8: Detection rate for each of the five individual failure modes._ The pattern closely tracks which failure modes are represented by a dedicated engineered feature: PWF (85.3%) and OSF (85.7%), both with bespoke features (power and overstrain_ratio), are detected far more reliably than TWF (37.0%) and HDF (33.9%), which have no equivalent. RNF's reported 31.6% detection rate should not be read as genuine signal: RNF is defined in the dataset as a 0.1% random failure probability, independent of any process parameter, so no feature-based approach can meaningfully detect it. At this contamination level, roughly a third of all observations are flagged, so RNF's detection rate is best explained by incidental overlap rather than the model having learned anything about random failure.

Taken together, these results indicate that Isolation Forest's effectiveness in this setting was driven substantially by how well each failure mechanism was represented in the feature space — a finding with direct implications for the Conclusions that follow.






## Conclusions:

Conclusions from the project findings and results.

## Next steps:  

With any analysis it is important to assess how the model and application of the analytical methods can be used and evolved to support the business goals and business decisions and yield tangible benefits.


## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/t.py)

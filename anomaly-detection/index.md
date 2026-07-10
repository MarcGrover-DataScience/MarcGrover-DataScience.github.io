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

Results from the project related to the business objective.

## Conclusions:

Conclusions from the project findings and results.

## Next steps:  

With any analysis it is important to assess how the model and application of the analytical methods can be used and evolved to support the business goals and business decisions and yield tangible benefits.


## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/t.py)

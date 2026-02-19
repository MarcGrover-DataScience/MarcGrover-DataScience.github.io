---

layout: default

title: E-Commerce Transactional Data (Association Rule)

permalink: /association-rule/

---

# This project is in development

## Goals and objectives:

The business goal of an e-commerce retailer is to generate and understand association rules for products, and uncover hidden patterns in customer purchasing behaviour and translate them into actionable commercial insights.  This demonstrates the practical application of Association Rule Learning on real-world transactional data to meet the goals and provide valuable insight to the business.

An objective is not simply to generate association rules, but to demonstrate the critical thinking required to validate their quality, understand their limitations, and prioritise those with genuine commercial relevance using metrics including lift, confidence, support, leverage, and conviction.  
 
A secondary objective is to illustrate how data science can directly inform business strategy in a retail context. The insights derived from this analysis have tangible applications across multiple commercial functions — from powering product recommendation engines and designing promotional bundles, to informing inventory planning and guiding targeted email marketing campaigns. 

By grounding every analytical decision in a business rationale, this project aims to demonstrate not only technical proficiency in Python, FP-Growth modelling, and data visualisation, but also the ability to communicate findings in a way that is meaningful to both technical and non-technical stakeholders. Ultimately, the project reflects a core principle of applied data science: that the value of a model lies not in its construction, but in the decisions it enables.

## Application:  

Details of how this is applicable to multiple industries to solve business problems, generate insight and provide tangible business benefits. 

Association Rule Learning is an unsupervised machine learning technique used to discover interesting relationships, patterns, and dependencies between variables in large datasets.  

At its core, the algorithm identifies "if-then" relationships — for example, if a customer buys product A, then they are likely to also buy product B. 

These rules are evaluated using three key metrics:  
* support (how frequently the itemset appears in the dataset)  
* confidence (the probability that the consequent occurs given the antecedent)  
* lift (how much more likely the association is compared to random chance)  

The most well-known algorithm for implementing this technique is the Apriori algorithm, though more efficient alternatives like FP-Growth are widely used in practice.  

In a real-world retail context, Association Rule Learning is the engine behind product recommendation engines and physical store layout optimisation. A supermarket chain, for example, could apply this technique to millions of transaction records to discover that customers who purchase nappies and baby formula on weekday evenings also frequently purchase beer.  This seemingly counterintuitive insight — famously observed in early retail analytics — could inform targeted promotions, shelf placement decisions, or personalised email campaigns.  

Beyond retail, the technique finds application in many other sectors:
* healthcare: identifying co-occurring symptoms or medications
* cybersecurity: detecting patterns in network intrusion events
* web analytics: understanding click-path behaviour across a site
* retail: detecting co-purchased items, supporting product bundling offers
* technology: understanding user behaviours to support on-boarding and retention
* science: in weather prediction, discover associations between atmospheric variables — such as sea surface temperature, humidity, and wind patterns — that tend to precede extreme weather events, contributing to improved early warning systems

## Methodology:  

Details of the methodology applied in the project.

Using the Online Retail II dataset — a genuine record of over 540k transactions from a UK-based e-commerce retailer — the project moves through the full analytical lifecycle: from raw data ingestion and rigorous preprocessing, through exploratory analysis and model development, to the interpretation and business contextualisation of results. 

https://www.kaggle.com/datasets/jillwang87/online-retail-ii?select=online_retail_10_11.csv

## Results:

Results from the project related to the business objective.

## Conclusions:

Conclusions from the project findings and results.

## Next steps:  

With any analysis it is important to assess how the model and application of the analytical methods can be used and evolved to support the business goals and business decisions and yield tangible benefits.


## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/test.py)

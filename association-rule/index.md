---

layout: default

title: E-Commerce Transactional Data (Association Rule)

permalink: /association-rule/

---

# This project is in development

## Goals and objectives:

For this portfolio project, the simulated business scenario is regarding a ficticious e-commerce retailer, with the goals of uncovering hidden patterns in customer purchasing behaviour and translate them into actionable commercial insights, enabling tangible business benefits.  This demonstrates the practical application of Association Rule Learning on real-world transactional data to meet the goals and provide valuable insight to the business, by understanding patterns of multiple products purchased together within the same transaction. 

An objective is not simply to generate association rules, but to demonstrate the critical thinking required to validate their quality, understand their limitations, and prioritise those with genuine commercial relevance using metrics including lift, confidence, support, leverage, and conviction.  
 
A secondary objective is to illustrate how data science can directly inform business strategy in a retail context. The insights derived from this analysis have tangible applications across multiple commercial functions — from powering product recommendation engines and designing promotional bundles, to informing inventory planning and guiding targeted email marketing campaigns. 

By grounding every analytical decision in a business rationale, this project aims to demonstrate not only technical proficiency in Python, FP-Growth modelling, and data visualisation, but also the ability to communicate findings in a way that is meaningful to both technical and non-technical stakeholders. Ultimately, the project reflects a core principle of applied data science: that the value of a model lies not in its construction, but in the decisions it enables.

## Application:  

Details of how this is applicable to multiple industries to solve business problems, generate insight and provide tangible business benefits. 

Association Rule Learning is an unsupervised machine learning technique used to discover interesting relationships, patterns, and dependencies between variables in large datasets.  

At its core, the algorithm identifies "if-then" relationships — for example, if a customer buys product A, then they are likely to also buy product B. 

These rules are evaluated using three key metrics:  
* **support**: how frequently the itemset appears in the dataset  
* **confidence**: the probability that the consequent occurs given the antecedent  
* **lift**: how much more likely the association is compared to random chance  

The most well-known algorithm for implementing this technique is the Apriori algorithm, though more efficient alternatives like FP-Growth are widely used in practice.  

In a real-world retail context, Association Rule Learning is the engine behind product recommendation engines and physical store layout optimisation. A supermarket chain, for example, could apply this technique to millions of transaction records to discover that customers who purchase nappies and baby formula on weekday evenings also frequently purchase beer.  This seemingly counterintuitive insight — famously observed in early retail analytics — could inform targeted promotions, shelf placement decisions, or personalised email campaigns.  

Beyond retail, the technique finds application in many other sectors:
* **healthcare**: identifying co-occurring symptoms or medications
* **cybersecurity**: detecting patterns in network intrusion events
* **web analytics**: understanding click-path behaviour across a site
* **retail**: detecting co-purchased items, supporting product bundling offers
* **technology**: understanding user behaviours to support on-boarding and retention
* **science**: in weather prediction, discover associations between atmospheric variables — such as sea surface temperature, humidity, and wind patterns — that tend to precede extreme weather events, contributing to improved early warning systems

## Methodology:  

Details of the methodology applied in the project.

This portfolio project uses the 'Online Retail II dataset', available at Kaggle [here](https://www.kaggle.com/datasets/jillwang87/online-retail-ii?select=online_retail_10_11.csv)  This is a genuine record of over 540k transactions from a UK-based e-commerce retailer — the project moves through the full analytical lifecycle: from raw data ingestion and rigorous preprocessing, through exploratory analysis and model development, to the interpretation and business contextualisation of results. 

The methodology follows the end-to-end data science workflow, implemented in Python using the mlxtend, pandas, seaborn, and numpy libraries, progressing from raw data ingestion through to the extraction and communication of business insight across eight structured stages.

**Data Loading and Initial Exploration**: The dataset is loaded from CSV, with an initial review of shape, data types, descriptive statistics, and missing value counts conducted to establish a baseline understanding of data quality.  
**Data Validation and Pre-Processing**: Seven sequential cleaning steps are applied to remove records associated to cancelled orders, missing Customer IDs, negative and zero quantities, invalid prices, non-product stock codes, non-UK transactions, and single-item invoices. Row counts are logged at each step to maintain full transparency over the impact of each decision.  
**Exploratory Data Analysis**: Visualisations are produced building a rounded picture of customer behaviour before modelling begins.  covering top-selling products, monthly transaction volumes, day-of-week purchasing patterns, monthly revenue, and basket size distribution
**Basket Construction**: The cleaned transaction data is pivoted into a binary invoice-by-product matrix, where each cell is encoded as True or False to indicate whether a given product was purchased within that transaction, producing the input format required by the FP-Growth algorithm.  
**Frequent Itemset Mining**: The FP-Growth algorithm is applied to the basket matrix with a minimum support threshold of 2%, chosen through sensitivity analysis as a pragmatic balance between discovery breadth and result quality, to identify all product combinations appearing frequently enough across transactions to be commercially meaningful.  
**Association Rule Generation**: Rules are derived from the frequent itemsets with a minimum lift of 1.5 and minimum confidence of 0.20, with all rules enriched with support, confidence, lift, leverage, and conviction metrics to enable multi-dimensional evaluation of rule strength.  
**Model Validation**: Six validation checks are applied covering trivial rule filtering, leverage confirmation, conviction scoring, support stability, reciprocal rule detection, and a parameter sensitivity analysis across five support thresholds, collectively confirming that the retained rules represent genuine and robust product associations rather than statistical artefacts.  
**Business Insight Extraction and Visualisation**: The strongest rules are surfaced and visualised through rankings by lift and confidence, a lift distribution plot, a top antecedents chart, and a product co-occurrence heatmap, with each output interpreted in terms of its direct application to retail use cases including product recommendations, promotional bundling, and inventory planning.  

## Results:

Results from the project related to the business objective.

**Data Validation and Pre-Processing**
The data was pre-processed to remove records deemed as not adding analytical value, or potentially liable to produce misleading or incorrect results.  In a real-wrold scenario, this is subject to many factors including; business objectives, analysis goals and constraints and features or issues with the data.  For example, records associated to non-UK purchases were excluded from this analysis, but could be included given a different business scoping or goal.  One subtle rule applied that is of note, is that invoices containing a single purchased product were excluded - as this analysis finds relationships between multiple products in the same invoice, invoices containing a single product offer no analytical value.

The result of the data validation and pre-processing step is a dataset for analysis summarised as:

* Records              : 352,765
* Unique invoices      : 15,365
* Unique products      : 3,821
* Unique customers     : 3,819

## Conclusions:

Conclusions from the project findings and results.

## Next steps:  

With any analysis it is important to assess how the model and application of the analytical methods can be used and evolved to support the business goals and business decisions and yield tangible benefits.


## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/test.py)

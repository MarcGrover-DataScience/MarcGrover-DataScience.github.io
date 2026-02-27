---

layout: default

title: Project (Causal Impact Analysis)

permalink: /causal-impact-analysis/

---

# This project is in development

## Goals and objectives:

For this portfolio project, the simulated business scenario concerns a European retail pharmacy chain operating across multiple store locations, with the goal of measuring the true causal effect of a promotional campaign on store sales — isolating the intervention's impact from underlying trends and seasonal patterns that would otherwise confound a simple before-and-after comparison. This demonstrates the practical application of Causal Impact Analysis to real-world retail sales data, providing the business with a rigorous and defensible estimate of whether the promotion generated a genuine uplift in revenue, and if so, by how much.  

A key objective is to demonstrate that measuring the effect of an intervention requires more than comparing sales before and after the event. By constructing a synthetic counterfactual — a statistically modelled estimate of what sales would have looked like had the promotion never occurred — the analysis separates the true causal effect of the campaign from external factors such as seasonal trading patterns, broader market trends, and day-of-week variation. Control stores that did not receive the promotion are used to anchor this counterfactual, ensuring the comparison is both meaningful and statistically grounded.  

A secondary objective is to illustrate how Causal Impact Analysis can directly inform commercial decision-making in a retail context. The insights derived from this analysis have tangible applications across multiple business functions — from evaluating the return on investment of promotional spend, to providing the evidence base needed to decide whether a campaign warrants a broader rollout across additional stores or regions. By grounding every analytical decision in a clear business rationale, this project aims to demonstrate not only technical proficiency in Python and Bayesian structural time series modelling, but also the ability to frame and communicate causal questions in a way that is meaningful to both technical and non-technical stakeholders. Ultimately, the project reflects a core principle of applied data science: that understanding what caused an outcome is far more valuable to a business than simply observing that it occurred.

## Application:  

Causal Impact Analysis is a statistical technique used to estimate the causal effect of an intervention or event on a time series outcome — answering not just what happened, but what would have happened had the intervention never occurred. 

Developed by Google researchers, the method uses Bayesian structural time series models to construct a synthetic counterfactual: a prediction of how the target metric would have evolved in the absence of the intervention, based on its own historical behaviour and a set of related control variables that were themselves unaffected by the event.  The difference between the observed outcome and this counterfactual is then interpreted as the causal effect of the intervention.

This technique is particularly powerful in scenarios where randomised controlled trials are impractical or impossible.  Consider a business that launches a regional marketing campaign in one city while other cities continue as normal.  Causal Impact Analysis can use the untreated cities as control variables to model what sales in the target city would have looked like without the campaign, then quantify the lift attributable to it — complete with credible intervals that convey the uncertainty in that estimate.  

This approach is equally applicable across a wide range of domains - as its strength lies in producing a rigorous, interpretable causal narrative from observational data alone: 
* **technology**: measuring the revenue impact of a product feature release
* **healthcare**: assessing the effect of a public health intervention on hospital admissions
* **retail**: impact of a new loyalty programme in a subset of stores to measure its true effect on weekly spend per customer 
* **science**: assess the effect of conservation policy changes — such as the designation of a marine protected area — on fish population metrics over time

## Methodology:  

Details of the methodology applied in the project.

This portfolio project uses the ‘Rossmann Store Sales dataset’ (both files: train.csv and store.csv), available at Kaggle [here](https://www.kaggle.com/datasets/pratyushakar/rossmann-store-sales)   

## Results:

Results from the project related to the business objective.

## Conclusions:

Conclusions from the project findings and results.

## Next steps:  

With any analysis it is important to assess how the model and application of the analytical methods can be used and evolved to support the business goals and business decisions and yield tangible benefits.


## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/test.py)

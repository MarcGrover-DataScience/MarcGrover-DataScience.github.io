---

layout: default

title: Project (Causal Impact Analysis)

permalink: /causal-impact-analysis/

---

# This project is in development

## Goals and objectives:

The business objective is ...

## Application:  

Causal Impact Analysis is a statistical technique used to estimate the causal effect of an intervention or event on a time series outcome — answering not just what happened, but what would have happened had the intervention never occurred. 

Developed by Google researchers, the method uses Bayesian structural time series models to construct a synthetic counterfactual: a prediction of how the target metric would have evolved in the absence of the intervention, based on its own historical behaviour and a set of related control variables that were themselves unaffected by the event.  The difference between the observed outcome and this counterfactual is then interpreted as the causal effect of the intervention.

This technique is particularly powerful in scenarios where randomised controlled trials are impractical or impossible.  Consider a business that launches a regional marketing campaign in one city while other cities continue as normal.  Causal Impact Analysis can use the untreated cities as control variables to model what sales in the target city would have looked like without the campaign, then quantify the lift attributable to it — complete with credible intervals that convey the uncertainty in that estimate.  

This approach is equally applicable across a wide range of domains - Its strength lies in producing a rigorous, interpretable causal narrative from observational data alone: 
* measuring the revenue impact of a product feature release (technology)
* assessing the effect of a public health intervention on hospital admissions (healthcare)
* impact of a new loyalty programme in a subset of stores to measure its true effect on weekly spend per customer (retail)
* assess the effect of conservation policy changes — such as the designation of a marine protected area — on fish population metrics over time (science)

## Methodology:  

Details of the methodology applied in the project.

## Results:

Results from the project related to the business objective.

## Conclusions:

Conclusions from the project findings and results.

## Next steps:  

With any analysis it is important to assess how the model and application of the analytical methods can be used and evolved to support the business goals and business decisions and yield tangible benefits.


## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/test.py)

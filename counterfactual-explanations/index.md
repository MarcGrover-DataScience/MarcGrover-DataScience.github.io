---

layout: default

title: The Minimal Path to a Different Outcome (Counterfactual Explanations)

permalink: /counterfactual-explanations/

---

# This project is in development

## Goals and objectives:

The [LIME project](/lime/) established that the MLP model's decisions rest on a strikingly narrow foundation: across four representative cases, `capital_gain` and `capital_loss` dominated every single explanation, at contribution magnitudes so consistent (+0.65 to +0.69 for `capital_gain`, regardless of which individual was being explained) that the finding looked less like ordinary feature importance and more like near-deterministic reliance on two features out of fourteen.

LIME can show that this reliance exists. It cannot show how far it goes. This project answers the natural follow-up question LIME's own Next Steps section raised: if `capital_gain`/`capital_loss` were locked at an individual's actual values and every *other* actionable feature were free to change, could the model's decision still be flipped — or is the reliance strong enough that no combination of the remaining features can substitute for it?

Reusing the same trained MLP, fitted preprocessor, and test-set artifacts exported by the MLP project — and explaining the identical four cases LIME selected, for direct continuity — the objectives were to:

- **Find the minimal, actionable change to an individual's features that would flip the model's prediction**, using DiCE (Diverse Counterfactual Explanations) as the primary search method, restricted to features an individual could plausibly act on.
- **Directly stress-test the LIME finding** by searching under two regimes for every case: once with `capital_gain`/`capital_loss` free to vary, and once with them locked at the individual's actual values — turning "these features dominate the explanation" into a falsifiable, quantified claim about whether the model can be flipped *without* them at all.
- **Cross-validate DiCE's proposals with an independent, exactly-computable method** — a manual single-axis search along `capital_gain` alone — rather than relying on a single search algorithm's output.
- **Treat "no counterfactual found" as a legitimate result**, not a failure to hide, wherever the locked regime genuinely cannot flip a case within a realistic search budget.

## Application:  

Details of how this is applicable to multiple industries to solve business problems, generate insight and provide tangible business benefits. 


## Methodology:  

Counterfactual explanations address a different and complementary question to most explainability techniques: not "why did the model predict this?", but "what would need to change for the model to predict something else?" A counterfactual explanation identifies the smallest, most plausible change to an individual's input features that would flip the model's prediction to a different, typically more favourable, outcome.

The core principle is one of minimal, actionable perturbation. Given a specific observation and an undesired prediction, a counterfactual search algorithm explores the feature space to find a nearby point — one differing from the original as little as possible, and ideally involving only features the individual could realistically influence — that the model would classify differently. Rather than describing the model's internal reasoning, this approach describes a concrete path from the current outcome to a desired one, framed in terms the affected individual can act upon. Constraints are typically applied during the search to ensure the resulting counterfactual is both plausible, remaining within the realistic bounds of the training data distribution, and actionable, avoiding changes to immutable characteristics such as age or protected attributes.

This distinguishes counterfactual explanations from feature-attribution methods such as LIME or SHAP: rather than ranking the importance of features in a prediction, they generate a specific, alternative scenario. This makes them particularly well suited to contexts where the end user is the individual affected by the decision, rather than the analyst building the model.

This approach is applicable across many sectors and scenarios. Practical examples showing where Counterfactual Explanations provide clear business value include:

🏦 **Finance**:

**Credit decision recourse**: A lender tells a declined loan applicant that approval would have followed if their credit utilisation had been reduced by a specific amount, giving the applicant a concrete, achievable path to future approval rather than a vague rejection.

**Insurance premium guidance**: An insurer explains to a policyholder the specific behavioural or coverage changes that would reduce their premium, supporting transparent and actionable customer communication.

**Regulatory compliance**: Financial institutions use counterfactual explanations to demonstrate to regulators that adverse automated decisions are based on legitimate, actionable factors rather than protected characteristics.

🏥 **Healthcare**:

**Treatment pathway planning**: Clinicians explore which modifiable patient factors — such as blood pressure or medication adherence — would need to change to move a patient out of a high-risk prediction band, informing care planning discussions.

**Preventative care targeting**: Public health teams identify the smallest lifestyle interventions that would shift an individual out of a predicted high-risk category, focusing preventative resources on the most impactful, achievable changes.

**Clinical trial eligibility**: Researchers determine the minimal changes in patient biomarkers that would alter eligibility classification for a trial, supporting recruitment strategy.

👥 **Human Resources**:

**Hiring feedback**: Recruiters give rejected candidates specific, constructive feedback on which qualifications or experience would have changed an automated screening outcome, improving candidate experience and reducing legal exposure.

**Promotion pathway clarity**: Employees are shown the specific, achievable performance changes that would alter a predicted promotion-readiness classification, supporting transparent career development.

**Retention intervention design**: HR teams identify the minimal, actionable changes — such as workload adjustment — that would shift an employee out of a high flight-risk prediction, informing targeted retention conversations.

🛍️ **Retail & Marketing**:

**Customer retention offers**: A subscription business identifies the smallest incentive or service change that would flip a customer's predicted churn outcome, informing cost-effective retention offer design.

**Loan and credit product eligibility**: A retail finance provider shows a customer declined for a store credit product the specific change in spending behaviour that would alter their eligibility outcome.

**Marketing eligibility transparency**: Customers excluded from a loyalty tier are shown the specific, achievable spending threshold that would include them, supporting transparent programme design.

## Results:

Results from the project related to the business objective.

## Conclusions:

Conclusions from the project findings and results.

## Next steps:  

Next steps based on current results and conclusions from above and suggested follow-up actions, analysis etc.

## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/conterfactual_explanations_v1.2.py)

---

layout: default

title: Project (Feedforward Neural Network)

permalink: /feedforward-neural-network/

---

# This project is in development

## Goals and objectives:

The business objective is ...

## Application:  

The Feedforward Neural Network — more commonly known in its simplest form as a Multi-Layer Perceptron (MLP) — is the foundational architecture of modern deep learning, and remains one of the most widely deployed supervised learning techniques in production systems today. It is described as "feedforward" because information flows in a single direction, from the input layer through one or more hidden layers to the output layer, with no cycles or feedback loops — as distinct from recurrent architectures, which pass information back through the network across sequential steps. Despite the emergence of far more specialised architectures for images, sequences, and language, the plain MLP remains the workhorse of choice wherever the input data is naturally tabular or already exists as a fixed-length feature vector — precisely the structure of the Adult Income dataset used in this project.

The core mechanism behind an MLP is straightforward to describe and surprisingly powerful in practice. Each layer consists of a set of neurons, and every neuron computes a weighted sum of its inputs, adds a bias term, and passes the result through a non-linear activation function — commonly ReLU (Rectified Linear Unit) for hidden layers, and sigmoid or softmax for the output layer in classification tasks. It is this non-linearity, applied repeatedly across layers, that gives the network its expressive power: a network with even a single hidden layer of sufficient width can, in principle, approximate any continuous function to arbitrary precision — a result known as the Universal Approximation Theorem. In practice, deeper networks with modest layer widths tend to learn more efficiently than a single very wide layer, because each layer builds progressively more abstract representations on top of the last. Training proceeds via backpropagation: the network's predictions are compared against known outcomes using a loss function, and the resulting error is propagated backwards through the network to update every weight in proportion to its contribution to that error, typically using a gradient-based optimiser such as Adam or stochastic gradient descent, repeated over many passes (epochs) through the training data.
This versatility means MLPs are deployed wherever an organisation needs to learn a complex, non-linear relationship between a set of measured or recorded features and an outcome of interest. Practical examples showing where the Feedforward Neural Network technique provides clear business value include:

🏦 **Finance & Insurance**:

**Credit Scoring and Loan Default Prediction**: Retail and commercial lenders use MLPs to predict the probability that an applicant will default on a loan, learning non-linear interactions between income, credit history, employment status, and existing debt that simpler linear scoring models struggle to capture — directly analogous to the income prediction task in this project.

**Insurance Claims Fraud Detection**: Insurers train feedforward networks on historical claims data — claim amount, policy tenure, claimant history, incident description features — to flag applications with an elevated probability of fraud for manual review, reducing the manual caseload while preserving coverage of genuinely suspicious claims.

**Algorithmic Trading Signals**: Quantitative trading desks use MLPs to combine dozens of engineered market indicators into a single predictive signal, exploiting the network's capacity to learn interaction effects between indicators that no single technical rule would capture on its own.

🏥 **Healthcare & Life Sciences**:

**Patient Risk Stratification**: Hospitals and insurers use MLPs trained on structured patient records — vitals, lab results, comorbidities, prior admissions — to predict the risk of an adverse outcome such as readmission or deterioration, supporting earlier clinical intervention for the highest-risk patients.

**Diagnostic Support from Structured Clinical Data**: Where a diagnosis depends on combining many individually weak clinical indicators — blood panel results, demographic factors, symptom checklists — MLPs are used to learn the combined, non-linear signal that a manual scoring rubric would miss.

**Drug Dosage Optimisation**: Pharmacokinetic models built as MLPs predict an individual patient's likely response to a given drug dosage from their physiological characteristics, supporting personalised dosing decisions beyond fixed, one-size-fits-all guidelines.

🛒 **Retail & Marketing**:

**Customer Churn Prediction**: Subscription businesses train MLPs on customer behaviour, usage, and billing history to predict the likelihood of cancellation, enabling retention teams to prioritise outreach to the customers most likely to leave and most likely to respond to intervention.

**Demand Forecasting**: Retailers use feedforward networks to predict product-level demand from a combination of pricing, seasonality, promotional activity, and historical sales features, supporting more accurate inventory and staffing decisions than simpler linear forecasting methods.

**Customer Lifetime Value Estimation**: Marketing teams model the expected long-term value of a newly acquired customer from early behavioural signals, allowing acquisition spend to be allocated toward the channels and segments most likely to yield high-value customers.

🏭 **Operations & Manufacturing**:

**Predictive Maintenance**: Manufacturers train MLPs on sensor readings — vibration, temperature, pressure, run-time — to predict the probability of equipment failure within a defined window, allowing maintenance to be scheduled proactively rather than reactively.

**Employee Attrition Modelling**: HR analytics teams use MLPs on structured workforce data — tenure, compensation, role, engagement survey results — to identify employees at elevated risk of leaving, supporting targeted retention conversations before a resignation is submitted.

**Supply Chain Risk Scoring**: Logistics and procurement teams score suppliers or shipment routes for disruption risk using structured operational and historical performance features, learning the non-linear combinations of factors that most reliably precede a delay or failure.

Across every one of these examples, the appeal of the MLP is the same: it removes the need to manually specify which combinations and interactions of features matter, learning that structure directly from labelled examples instead. This makes it a natural fit for exactly the kind of problem this project addresses — predicting an individual-level outcome from a wide set of demographic and behavioural features — while the practical challenges tackled here (class imbalance, feature scaling, encoding, and CPU-only training constraints) are the same ones any organisation deploying this technique on real, imperfect data has to solve.


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
[View the Python Script](/t.py)

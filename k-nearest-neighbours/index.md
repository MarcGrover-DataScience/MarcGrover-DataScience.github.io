---

layout: default

title: Project (K-Nearest Neighbours)

permalink: /k-nearest-neighbours/

---

# This project is in development

## Goals and objectives:

For this portfolio project, the business scenario concerns the prediction of red wine quality from physicochemical measurements — a classification problem drawn from the UCI Wine Quality dataset, widely used as a benchmark in applied machine learning. The dataset comprises 1,599 observations of red wine, with eleven continuous chemical measurements including alcohol content, acidity, sulphates, and density, alongside a quality rating assigned by human sensory panels on a scale of three to nine. The objective is to determine whether the quality of a wine can be reliably predicted from its chemical composition alone, using K-Nearest Neighbours (KNN) as the classification framework.

KNN is the appropriate technique for this problem for several reasons. As a distance-based algorithm, it makes no assumption about the functional form of the relationship between features and the target — an important property here, because wine quality is determined by the complex, non-linear interaction of multiple chemical variables rather than by any single dominant measurement. The algorithm classifies a new observation by identifying its K closest neighbours in feature space and assigning the majority class among them, a logic that maps naturally onto the intuition that wines with similar chemical profiles tend to be rated similarly by tasters.

A key design decision in the analysis is the binning of raw quality scores into three ordered classes — Low (scores 3–4), Medium (scores 5–6), and High (scores 7–9). This is motivated by the heavy concentration of scores in the middle of the scale, which would render direct prediction of individual integer scores impractical and would obscure meaningful quality distinctions. The three-class formulation retains the ordinal structure of the problem while producing a classification task that is both analytically tractable and practically interpretable.

A secondary objective is the selection of the optimal value of K through systematic evaluation across a range of candidate values. The choice of K is the primary hyperparameter in KNN and directly governs the bias-variance trade-off of the model: small values of K produce highly flexible boundaries susceptible to overfitting, while large values produce smoother boundaries at the cost of failing to capture local structure in the data. By evaluating model performance across K = 1 to K = 30 on held-out test data, the analysis identifies the value that maximises generalisation accuracy and demonstrates the importance of principled hyperparameter selection.

By the end of the analysis, the project aims to demonstrate not only the correct implementation of KNN classification — including mandatory feature scaling, stratified train-test splitting, and model evaluation — but also the analytical judgement to interpret results in terms of practical significance, and the ability to communicate the findings of a multi-feature classification model clearly to both technical and non-technical audiences.

## Application:  

K-Nearest Neighbours is a versatile, interpretable non-parametric supervised machine learning classification algorithm deployed across a wide range of business domains wherever the goal is to assign observations to categories based on their proximity to known examples in a multi-dimensional feature space.  It is used for both classification and regression tasks.

The core principle behind KNN is that data points with similar features exist in close proximity to each other in vector space. When presented with a new, unseen data point, the algorithm calculates the "distance" (typically using Euclidean, Manhattan, or Minkowski metrics) between that point and all other points in the training dataset. It then identifies the $k$ closest data points—the "nearest neighbours"—and assigns the new point the most common class among them (for classification) or the average of their values (for regression). As a "lazy learner," KNN does not build an explicit internal model during a training phase, but rather stores the entire dataset and performs the computation only when a prediction is required.  

This approach is applicable across many sectors and scenarios. Practical examples showing where the K-Nearest Neighbours technique provides clear business value include: 

🛍️ **Retail**:  

**Product Recommendations**: An e-commerce platform suggests new items to a user by identifying the $k$ customers with the most similar browsing and purchasing histories and recommending what they bought.  
**Customer Segmentation**: Marketing teams classify new customers into specific promotional tiers based on how closely their demographic and spending profiles match existing, well-defined customer groups.  
**Store Placement**: A retail chain predicts the potential profitability of a proposed new store location by averaging the historical revenue of the $k$ existing stores with the most similar local demographic and economic indicators.

💻 **Technology**:

**Intrusion Detection**: Cybersecurity systems flag network traffic as a potential cyberattack if its packet characteristics are closest to known historical malicious signatures rather than normal baseline traffic.  
**Optical Character Recognition (OCR)**: Document scanning software classifies handwritten letters by converting the image pixels into a vector and finding the closest matching confirmed characters in its database.  
**Content Curation**: Streaming services dynamically predict user ratings for a new movie by analyzing the ratings given to that same movie by the $k$ users who have the most similar overall viewing tastes.

🔬 **Science & Research**:

**Genetics**: Biologists classify the function of newly discovered genetic sequences by finding the most structurally similar known sequences within a vast DNA database.  
**Drug Discovery**: Pharmaceutical researchers predict the potential toxicity of a new chemical compound based on the known toxicological properties of its nearest structural neighbours.  
**Geospatial Estimation**: Environmental scientists estimate missing soil moisture data for a specific mapping grid by taking the weighted average of the readings from the $k$ closest geographic sensor stations.

🏭 **Manufacturing**:

**Predictive Maintenance**: Factory systems predict impending machine failures by comparing real-time vibration and temperature sensor data to the nearest matching historical patterns that preceded equipment breakdowns.  
**Quality Control**: Computer vision systems classify manufactured components on an assembly line as either acceptable or defective based on their dimensional similarity to a training set of perfectly engineered parts.  
**Supply Chain Routing**: Logistics software estimates the delivery time for a new shipment by averaging the transit times of the $k$ most similar historical shipments in terms of distance, weight, and weather conditions.

## Methodology:  

The methodology adopted for this project follows the end-to-end data science workflow, progressing from data loading and validation through exploratory analysis, pre-processing, model fitting, and evaluation. The project is implemented in Python, using pandas for data manipulation, scikit-learn for modelling and evaluation, and seaborn and matplotlib for visualisation. Each stage of the pipeline is described below.

**Data Loading and Validation**:  

The dataset is loaded from the locally downloaded Kaggle CSV file using pandas, specifying the semicolon delimiter used in the UCI Wine Quality format. A structured validation audit is conducted prior to any analysis, checking for missing values across all eleven feature columns and the quality target, identifying and removing duplicate records, and confirming that all columns carry the expected numeric data types. Descriptive statistics are printed for all variables, and the raw distribution of quality scores is inspected to confirm the spread of ratings and motivate the subsequent banding decision.

**Feature Engineering — Quality Banding**:

The raw quality scores are binned into three ordered classes: Low (scores 3–4), Medium (scores 5–6), and High (scores 7–9). This decision is driven by the empirical distribution of scores in the red wine dataset, where the overwhelming majority of observations carry scores of five or six, and very few sit at the extremes. Retaining the raw integer scores as a nine-class target would result in severe class imbalance and an effectively unlearnable problem for the minority classes. The three-class formulation preserves the ordinal quality distinction that is meaningful to a wine producer or buyer while producing a balanced enough target for a fair classification evaluation.

**Exploratory Data Analysis**:

Exploratory analysis is conducted to characterise the distribution of quality bands and to understand how the individual physicochemical features relate to quality. The following charts are produced:

* A **bar chart** of quality band counts, confirming the class distribution following banding and validating that the three classes are sufficiently represented for modelling.
* A **correlation heatmap** of the full feature matrix including the raw quality score, used to identify features with strong linear relationships to quality and to detect multicollinearity between predictors.
* **Five boxplots** — one each for alcohol, volatile acidity, sulphates, citric acid, and density — showing the distribution of each feature across the three quality bands. These features are selected on the basis of their correlation with quality and their chemical interpretability. The boxplots allow visual assessment of whether quality-related differences exist for individual features in isolation, before the KNN model is used to exploit multi-feature proximity.

**Pre-Processing**:

The feature matrix is separated from the target variable and split into training and test sets using an 80/20 ratio, with stratification on the quality band target to preserve class proportions in both sets. Feature scaling is applied using scikit-learn's StandardScaler, fitted on the training set and applied to both training and test sets. Scaling is a mandatory pre-processing step for KNN: because the algorithm computes Euclidean distances between observations, features measured on different scales — such as total sulphur dioxide (tens to hundreds) and pH (2.5 to 4.0) — would otherwise dominate the distance calculation purely by virtue of their numerical range, producing a distorted notion of similarity that the unscaled data does not support.

**Optimal K Selection**:

A KNN classifier is trained and evaluated for each integer value of K from 1 to 30. For each value, both training accuracy and test accuracy are recorded. The resulting accuracy curves are plotted against K, with the optimal value — defined as the K that maximises test accuracy — identified and marked. This step makes the bias-variance trade-off tangible: the chart typically shows high training accuracy and low test accuracy at very small K (overfitting), converging as K increases and stabilising at the optimal point before gradually declining again.

**Model Fitting and Evaluation**:

A final KNN classifier is trained using the optimal K on the full training set. Predictions are generated on the held-out test set and evaluated using three complementary outputs: a classification report providing per-class precision, recall, and F1-score; an overall test accuracy figure; and a confusion matrix heatmap showing the distribution of correct and incorrect predictions across the three quality bands. The confusion matrix is particularly informative for a multi-class problem, revealing not just the overall error rate but the pattern of misclassification — specifically whether the model tends to confuse adjacent quality bands (a more forgivable error) rather than assigning Low and High classifications incorrectly to one another.

**Permutation Feature Importance**:

Permutation feature importance is calculated using scikit-learn's permutation_importance function with 20 repeat permutations per feature on the test set. This method measures the decrease in model accuracy when each feature's values are randomly shuffled, isolating the contribution of each variable to the model's predictive power. It is model-agnostic and does not rely on internal model parameters, making it well-suited to KNN where no native feature importance measure exists. Results are presented as a ranked horizontal bar chart with standard deviation error bars, providing a stable and interpretable view of which chemical properties drive the model's classification decisions.

## Results:

Results from the project related to the business objective.

## Conclusions:

Conclusions from the project findings and results.

## Next steps:  

With any analysis it is important to assess how the model and application of the analytical methods can be used and evolved to support the business goals and business decisions and yield tangible benefits.


## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/t.py)

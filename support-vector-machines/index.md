---

layout: default

title: Project (Support Vector Machines)

permalink: /support-vector-machines/

---

# This project is in development

## Goals and objectives:

For this portfolio project, the business scenario concerns the classification of breast tumours as malignant or benign from digitised cell nucleus measurements — a binary classification problem drawn from the Wisconsin Breast Cancer Diagnostic dataset, available directly from scikit-learn. The dataset comprises 569 observations across 30 continuous features derived from digitised fine needle aspirate (FNA) images, including measurements of cell nucleus radius, texture, perimeter, area, and concavity. Full data validation and exploratory analysis for this dataset were conducted in the Decision Tree project and are not repeated here; the SVM analysis proceeds directly from that established foundation.

This is the fourth project in the portfolio to apply a classification algorithm to this dataset, following Decision Trees (93.86% accuracy), Random Forests (95.61%), and Gradient Boosted Trees (97.37%). The progressive accuracy improvements across those three projects reflect successive refinements within the same algorithmic family — tree-based ensemble methods. This project deliberately steps outside that family. Support Vector Machine (SVM) is a geometrically motivated classifier that operates on an entirely different principle: rather than building an ensemble of decision rules, it identifies the single optimal hyperplane that separates classes with the greatest possible margin. Applying SVM to the same dataset and evaluation framework as the prior three projects provides a direct, controlled basis for comparing not just accuracy figures but the analytical characteristics of two fundamentally different approaches to supervised classification.

A primary objective of the project is to demonstrate the kernel trick — SVM's mechanism for classifying data that is not linearly separable in its original feature space. To make this concept visually explicit, the project opens with a brief illustration using synthetically generated data (sklearn.datasets.make_circles), where two concentric classes are linearly inseparable by construction. A linear SVM applied to this data fails visibly; an RBF kernel SVM separates the classes correctly by implicitly mapping observations into a higher-dimensional space in which a separating hyperplane exists. This illustration is not the primary analysis — it serves as a clear, self-contained demonstration of the kernel trick before the same principles are applied to the Breast Cancer dataset, where the non-linearity is real but less visually apparent.

The principal hyperparameters governing SVM performance are the regularisation parameter **C** and, for the RBF kernel, the kernel coefficient **gamma**. C controls the trade-off between maximising the margin and minimising classification errors on the training data: small values of C enforce a wider margin at the cost of tolerating some misclassification, while large values force the boundary closer to the training observations in order to classify them correctly, at the risk of overfitting. Gamma determines the reach of individual training observations in shaping the decision boundary: small values of gamma produce smooth, broadly influenced boundaries, while large values cause the boundary to follow training observations tightly, again increasing overfitting risk. A secondary objective is the identification of optimal values for both parameters through a systematic grid search, making the C–gamma interaction and its effect on the bias-variance trade-off a central analytical theme of the project.

By the end of the analysis, the project aims to demonstrate the correct implementation of SVM classification — including the mandatory feature scaling that distance-based algorithms require, kernel selection, and joint hyperparameter optimisation via GridSearchCV — alongside the interpretive judgement to contextualise the results within the progression established by the prior three projects. A forward reference to a planned SHAP and Model Interpretability project is included in the Next Steps section, as the decision boundary structure of SVM raises specific and interesting questions about feature-level explanation that warrant dedicated treatment.


## Application:  

Support Vector Machine (SVM) is a powerful, versatile supervised machine learning algorithm used for both classification and regression tasks, deployed across a wide range of industries wherever the objective is to identify a decision boundary that best separates observations into distinct categories — or, in regression settings, to model a continuous target within a defined tolerance. It performs particularly well in high-dimensional feature spaces and in settings where the boundary between classes is complex or non-linear.

The core principle behind SVM is the identification of the optimal hyperplane — a decision boundary in feature space that separates classes with the greatest possible margin. The margin is defined as the perpendicular distance between the hyperplane and the nearest training observations from each class; these boundary observations are known as **support vectors**, and they are the only points in the dataset that directly determine the position and orientation of the decision boundary. Maximising this margin is the algorithm's training objective, and it produces a boundary that is geometrically as far as possible from both classes, making the classifier robust to new observations near the boundary. Where the data are not linearly separable in their original feature space, SVM applies a **kernel function** — such as the Radial Basis Function (RBF), polynomial, or sigmoid kernel — to map observations into a higher-dimensional space in which a separating hyperplane can be found. This kernel trick allows SVM to model highly non-linear class boundaries without explicitly computing the transformed feature space, keeping computation tractable even when the implied dimensionality is very large.

This approach is applicable across many sectors and scenarios. Practical examples showing where the Support Vector Machine technique provides clear business value include:

🏥 **Healthcare & Life Sciences**:

**Cancer Classification**: Clinical decision support systems classify tumour biopsies as malignant or benign based on cell morphology measurements, where SVM's ability to handle high-dimensional medical imaging data and find a wide-margin boundary between classes has made it a long-established benchmark in pathology research.  

**Drug Activity Prediction**: Pharmaceutical researchers use SVM to predict whether a candidate molecule will be biologically active against a given target, treating molecular descriptor vectors — representations of chemical structure — as the feature space and exploiting the kernel trick to capture non-linear structure-activity relationships.  

**Patient Readmission Risk**: Hospital systems score the likelihood of a patient being readmitted within thirty days of discharge, using clinical measurements and treatment history to identify high-risk cases for targeted early intervention.  

💻 **Technology & Cybersecurity**:  

**Spam and Phishing Detection**: Email filtering systems classify messages as spam or legitimate by representing each email as a high-dimensional vector of word frequencies — a setting where SVM's strong performance in high-dimensional, sparse feature spaces gives it a natural advantage over distance-based methods.  

**Facial Recognition**: Biometric authentication systems use SVM classifiers trained on facial feature vectors to verify identity, leveraging the algorithm's capacity to separate classes precisely even when the number of features is large relative to the number of training examples.  

**Malware Classification**: Cybersecurity platforms classify executable files as malicious or benign based on extracted behavioural and structural features, where the ability to find a robust margin between classes helps the classifier generalise to previously unseen malware variants.  

🔬 **Science & Research**:  

**Remote Sensing and Land Use Classification**: Satellite imagery analysis systems classify land cover types — urban, agricultural, forested, coastal — from multispectral pixel data, a high-dimensional classification problem in which SVM consistently produces competitive accuracy against more computationally expensive alternatives.  

**Protein Function Prediction**: Bioinformaticians classify protein sequences by biological function using structural and sequence-derived feature vectors, taking advantage of SVM's ability to operate in the very high-dimensional spaces that characterise genomic and proteomic data.  

**Fault Detection in Geophysics**: Seismological research systems classify seismic waveform signals as genuine geological events or instrumental noise, where the clear margin between class distributions in feature space makes SVM a natural fit for the binary decision.

🏭 **Manufacturing & Industry**:  

**Quality Inspection**: Computer vision systems on production lines classify manufactured components as conforming or defective based on image-derived dimensional and surface measurements, with the maximum-margin boundary ensuring that borderline components are handled conservatively.

**Predictive Maintenance**: Industrial monitoring systems classify equipment operating conditions as normal or pre-failure based on vibration, temperature, and acoustic sensor readings, where the support vector structure means the classifier's decision is directly interpretable in terms of the specific operating conditions that define the boundary between safe and at-risk states.  

**Energy Load Forecasting**: Utility companies use SVM regression — known as Support Vector Regression (SVR) — to forecast electricity demand by fitting a function that falls within a defined tolerance of the observed load data, providing a robust prediction that is less sensitive to outliers than ordinary least squares approaches.  


## Methodology:  

The methodology for this project is structured in two parts, each implemented as a separate Python script. The first is a brief, self-contained kernel trick illustration using synthetically generated data, designed to make the core principle of non-linear SVM classification visually explicit before the main analysis begins. The second is the full end-to-end SVM classification pipeline applied to the Wisconsin Breast Cancer Diagnostic dataset. Both scripts are implemented in Python, using pandas for data handling, scikit-learn for modelling and evaluation, and seaborn and matplotlib for visualisation.

**Part 1: Kernel Trick Illustration** (svm_kernel_illustration.py)

**Data Generation**:

Synthetic data is generated using scikit-learn's make_circles function, producing two concentric circular classes that are linearly inseparable by construction. A modest number of observations is used — sufficient to produce a clear visual result without computational overhead — with a controlled noise parameter to introduce realistic scatter around the circular boundaries. The two-dimensional feature space is retained deliberately, as the purpose of this script is visual: a two-dimensional dataset can be plotted directly, making the decision boundary produced by each classifier immediately interpretable.

**Linear SVM — Baseline**:

A linear kernel SVM is fitted to the generated data. The resulting decision boundary is plotted over a mesh grid covering the full feature space, with training observations overlaid and colour-coded by class. Because the two classes form concentric rings, the linear boundary is unable to separate them correctly, and the plot makes this failure explicit — the boundary bisects the feature space with a straight line that inevitably misclassifies observations from both classes.

**RBF Kernel SVM**:

An RBF kernel SVM is fitted to the same data using default hyperparameter values. The decision boundary is plotted in the same format as the linear case, allowing direct visual comparison. The RBF kernel implicitly maps observations into a higher-dimensional space in which the concentric classes become separable, and the resulting boundary curves correctly around the inner class. The two plots — linear failure and RBF success — are produced as individual saved figures and are presented side by side in the Results section of the write-up to illustrate the practical value of the kernel trick.

**Part 2: Breast Cancer Classification** (svm_breast_cancer.py)

**Data Loading**:

The Wisconsin Breast Cancer Diagnostic dataset is loaded directly from scikit-learn using load_breast_cancer(). The feature matrix and binary target variable — malignant (0) or benign (1) — are extracted. As noted in the Goals and Objectives section, full data validation and exploratory analysis for this dataset were conducted in the Decision Tree project and are not repeated here. A brief confirmation of dataset dimensions and class distribution is printed for reference.

The dataset is also available at Kaggle [here](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

**Pre-Processing — Feature Scaling**:

The feature matrix is split into training and test sets using an 80/20 ratio, with stratification on the target variable to preserve class proportions in both sets. Feature scaling is then applied using scikit-learn's StandardScaler, fitted on the training set and applied to both training and test sets. Scaling is a mandatory pre-processing step for SVM: the algorithm constructs its decision boundary by computing distances between observations and the hyperplane in feature space. Features measured on different numerical scales would distort these distances and produce a biased boundary — not because any feature is genuinely more important, but purely as an artefact of its measurement units. Standardisation ensures that every feature contributes to the distance calculation on an equal footing.

**Kernel Comparison — Linear vs RBF**:

Prior to hyperparameter tuning, a direct comparison of the linear and RBF kernels is conducted on the Breast Cancer data, using default values for C and gamma. A classification report and accuracy figure are recorded for each kernel. This step connects the kernel illustration from Part 1 to the real-world data, and provides a principled basis for selecting the RBF kernel as the focus of the subsequent hyperparameter search. The results are presented as a grouped bar chart comparing per-class F1-scores across the two kernels.

**Hyperparameter Tuning — Grid Search over C and gamma**:

A systematic grid search is conducted over a range of values for the regularisation parameter C and the RBF kernel coefficient gamma, using scikit-learn's GridSearchCV with five-fold stratified cross-validation. C and gamma are searched jointly rather than sequentially, as their effects on the decision boundary interact: a high C combined with a high gamma produces a tightly fitted, complex boundary prone to overfitting, while a low C combined with a low gamma produces an overly smooth boundary that may underfit. The grid spans multiple orders of magnitude for both parameters — for example, C ∈ {0.1, 1, 10, 100, 1000} and gamma ∈ {0.0001, 0.001, 0.01, 0.1, 1} — ensuring that both extremes of the bias-variance spectrum are evaluated. The cross-validation accuracy scores across the full C–gamma grid are visualised as a heatmap, providing an intuitive view of how the two parameters interact and where performance is maximised. The optimal combination is identified and reported.

**Model Fitting and Evaluation**:

A final SVM classifier is trained using the optimal C and gamma values on the full training set, with probability=True enabled to allow the generation of predicted class probabilities alongside hard classifications. Predictions are generated on the held-out test set and evaluated using four complementary outputs: a classification report providing per-class precision, recall, and F1-score; an overall test accuracy figure for direct comparison with the Decision Tree, Random Forest, and Gradient Boosted Tree results; a confusion matrix heatmap; and an ROC curve with AUC score. The ROC-AUC metric is included specifically to enable comparison with the GBT project, which reported an ROC-AUC of 0.9964, and to assess SVM's discriminative ability across all classification thresholds rather than at a single operating point.

**Support Vector Analysis**:

The number of support vectors identified by the fitted model is reported, broken down by class. Support vectors are the subset of training observations that lie on or within the margin boundary and directly determine the position of the decision hyperplane — all other training observations have no influence on the boundary whatsoever. Reporting the support vector count provides a concrete characterisation of the fitted model: a large number of support vectors relative to the training set size suggests the margin is narrow and the boundary complex, while a small number indicates a wide, well-defined margin with strong generalisation properties. This is a model characteristic with no direct equivalent in the tree-based projects and gives the SVM page a distinctive analytical element.

## Results:

Results from the project related to the business objective.

## Conclusions:

Conclusions from the project findings and results.

## Next steps:  

With any analysis it is important to assess how the model and application of the analytical methods can be used and evolved to support the business goals and business decisions and yield tangible benefits.


## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/DecisionTree_BreastCancer.py)

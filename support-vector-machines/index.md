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

💻 Technology & Cybersecurity:  

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

Details of the methodology applied in the project.

## Results:

Results from the project related to the business objective.

## Conclusions:

Conclusions from the project findings and results.

## Next steps:  

With any analysis it is important to assess how the model and application of the analytical methods can be used and evolved to support the business goals and business decisions and yield tangible benefits.


## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/DecisionTree_BreastCancer.py)

# Codex_Techno_2
**Wine Quality Prediction**
**Project Overview**

This project focuses on predicting the quality of wine based on its chemical properties, offering a real-world application of machine learning in viticulture. The dataset contains attributes such as acidity, sugar, pH, and alcohol, which are used to classify wine quality.
The project compares different machine learning classifiers — Random Forest, Stochastic Gradient Descent (SGD), and Support Vector Classifier (SVC) — to evaluate which model performs best in predicting wine quality.

**Dataset**

Source: Wine Quality Dataset (UCI Machine Learning Repository)
Rows: ~6,400
Columns: 12 (Chemical features + Quality label)
Target variable: Wine Quality (score 0–10)

**For better interpretation, quality scores are grouped into categories:**
Low Quality (≤ 5)
Medium Quality (= 6)
High Quality (≥ 7)

**Technologies Used**

**Python**
Pandas, NumPy → Data cleaning & manipulation
Matplotlib, Seaborn → Data visualization
Scikit-learn → Model building & evaluation

**Key Steps**

**Data Cleaning & Preprocessing**
Handling missing values
Converting quality into categories (Low/Medium/High)
Feature scaling (StandardScaler)
Exploratory Data Analysis (EDA)
Distribution of wine quality
Relationship between chemical features & quality
Correlation heatmap

**Model Building**
Random Forest Classifier
Stochastic Gradient Descent (SGD) Classifier
Support Vector Classifier (SVC)

**Model Evaluation**
Accuracy, precision, recall, and F1-score
Confusion matrix visualization
Feature importance analysis

**Visualizations**

Quality distribution bar chart
Alcohol vs Quality scatter plot
Correlation heatmap

**Feature importance (Random Forest)**
Confusion matrix heatmap

**Results**

Random Forest achieved the highest accuracy among the three models.
Alcohol and volatile acidity were found to be key indicators of wine quality.

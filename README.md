# # Heart Disease Prediction Using Machine Learning

## Introduction
In the present generation, most people are working hard and neglecting their health, leading to various health issues such as heart diseases and lung diseases. Heart disease has become one of the leading causes of mortality in both men and women. The application of machine learning techniques in medical examinations can significantly enhance the accuracy of disease prediction. This study explores how machine learning and deep learning techniques can be used to predict heart diseases.

## Problem Definition
The goal of this study is to predict whether an individual is at risk of being diagnosed with heart disease, requiring quick detection and management. Machine learning and deep learning provide effective solutions by analyzing patient data and identifying those at higher risk. Various classification models in machine learning help to classify cardiac diseases efficiently.

## Context and Background
### Relevant Studies:
1. **Heart Disease Prediction Using Machine Learning: A Systematic Review** – K. S. Rajasekaran
2. **Predictive Modeling of Heart Disease Using Logistic Regression and Decision Trees** – P. R. Pustokhina
3. **UCI Data Set on Heart Disease** – M. M. R. D. W. T. Lichman
4. **Predicting Heart Disease Using Support Vector Machines** – S. S. Ghosh
5. **Ensemble Learning Algorithms for Heart Disease Prediction** – J. F. Alcaraz
6. **A Review of the Naïve Bayes Classifier for Medical Diagnosis** – S. Pradeep
7. **Evaluation of Machine Learning Methods for Diagnosing Heart Disease** – H. T. Dang
8. **Predicting Heart Disease Using Hybrid Machine Learning Models** – S. M. A. Tabesh
9. **Feature Selection Techniques in Machine Learning with Applications to Health Data** – H. S. Saeed

## Objectives and Goals
- Develop and assess machine learning models for heart disease prediction.
- Explore classification techniques such as SVM, KNN, ANN, Decision Trees, and Random Forest.
- Utilize optimization methods like Grid Search and Random Search.
- Preprocess and clean the dataset for enhanced model performance.
- Ensure transparency in healthcare applications, particularly for heart disease predictions.

## Summary of Approach
This research employs machine learning and deep learning techniques to predict cardiovascular diseases. Feature selection methods are used to improve model accuracy. Classification models such as ANN, SVM, KNN, Decision Trees, and Random Forest are trained and evaluated using performance metrics like precision, accuracy, recall, F1-score, and ROC-AUC. Data preprocessing techniques like numerical scaling and categorical encoding enhance dataset quality.

## Methods
### Data Acquisition and Sources
- The dataset contains **918 records** with **12 attributes** such as Age, Sex, Chest Pain, Resting BP, Cholesterol, and MaxHR.
- The target variable is binary: **1 (Heart Disease Present)**, **0 (No Heart Disease)**.
- Dataset obtained from Kaggle: [Kaggle Dataset Link](https://www.kaggle.com/code/sisharaneranjana/machine-learning-to-the-fore-to-save-lives/notebook)

### Mathematical and Statistical Methods
#### 1. Naïve Bayes
- Based on **Bayes Theorem**: 
  \[ P(X \mid Y) = \frac{P(Y)P(Y \mid X)}{P(X)} \]
- Predicts heart disease based on cholesterol, blood pressure, and age.

#### 2. Logistic Regression
- Used for binary classification (Heart Disease: Yes/No).
- Considers age, gender, BMI, and family history.

#### 3. K-Nearest Neighbors (KNN)
- Uses distance metrics for classification:
  - Euclidean Distance: \( d(a,b) = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2} \)
  - Manhattan Distance: \( d(a,b) = \sum_{i=1}^{n} |a_i - b_i| \)

#### 4. Artificial Neural Networks (ANN)
- Modeled using connected layers of neurons:
  - \( z = \sum_{i=1}^{n} w_i y_i + b \)
  - \( a = f(z) \)

#### 5. Random Forest
- Uses multiple decision trees for classification.
- Follows **bagging principle** for improved accuracy and reduced variance.

### Experimental Design
#### 1. Data Preprocessing
- Splitting dataset into **training and testing** sets.
- Converting categorical features into numerical values.
- Handling missing values and outliers.

#### 2. Feature Selection
- Techniques: **Chi-square test, Recursive Feature Elimination**.
- Key attributes: **Decision Tree, Random Forest**.

#### 3. Model Training and Evaluation
- Metrics: **F1-score, accuracy, precision, recall, ROC-AUC**.
- Training models: **KNN, Decision Trees, SVM, ANN, Random Forest**.

#### 4. Model Comparison and Selection
- Comparing accuracy and efficiency of different models.
- Selecting the best model for heart disease prediction.

## Software and Tools
- **Python**
- **Google Colab / Jupyter Notebook**
- **NumPy, Pandas**
- **Matplotlib, Seaborn**
- **Scikit-learn, TensorFlow/Keras**

---
This document outlines the heart disease prediction model using machine learning techniques and statistical methods. Further implementation will involve coding these models in Python, training them with real-world datasets, and analyzing their performance using various metrics.

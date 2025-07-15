# Medication-Effectiveness-Analysis-using-Machine-Learning
Machine learning project for predicting 30-day hospital readmissions in diabetes patients, focusing on medication effectiveness and clinical data analysis.

# 📊 Project Overview
Hospital readmissions are a major concern in healthcare, particularly for chronic diseases like diabetes. This project aims to predict 30-day hospital readmissions in diabetes patients by analyzing various factors, especially the effectiveness of prescribed medications.By building multiple machine learning models, this study provides insights into patient care improvement and hospital resource management.

# Problem Statement
The primary objective is to develop a binary classification model that predicts whether a diabetic patient will be readmitted within 30 days after discharge, based on patient details, medical history, medications, and hospital encounters.

# 🗃️ Dataset
• Source: Diabetic Data (UCI Machine Learning Repository)

• Records: 101,766 hospital admissions

• Features: 50+ attributes including demographics, diagnoses, medications, lab      results, etc.

# 🚀 Project Pipeline
1️⃣ **Data Preprocessing**

• Handling missing values

• Encoding categorical variables

• Outlier detection and removal

• Balancing dataset using SMOTE

2️⃣ **Exploratory Data Analysis (EDA)**

• Univariate and bivariate analysis

• Visualizations of medication impact on readmission

3️⃣ **Model Building & Evaluation**

**Models used:** Logistic Regression,Decision Tree,Random Forest,XGBoost,Stacking Ensemble

**Evaluation Metrics**: Accuracy,Precision,Recall,F1-Score,ROC-AUC Curve,Confusion Matrix

4️⃣ **Hyperparameter Tuning**

• Performed using Grid Search for optimal model performance

5️⃣ **Feature Importance**

• Visualized using XGBoost Feature Importance plots

## Results & Insights

• Best Model: XGBoost Classifier

• Achieved a balanced F1-Score, indicating effective prediction of minority class (readmitted patients).

• Stacking Ensemble improved generalization by combining multiple models.

• Medication types, number of prior admissions, and discharge disposition were among the most important features affecting readmission.

## Limitations

• Dataset may contain outdated or hospital-specific coding practices.

• Imbalanced classes required heavy use of balancing techniques, which may affect real-world generalization.

• No time-series analysis was performed (data was static).

• Medication adherence outside hospital visits is unknown.

## Future Work

• Incorporate deep learning models (LSTM for sequential data)

• Use real-time patient monitoring data

• Deploy the model using Flask or Streamlit for clinical decision support

• Integrate SHAP values for enhanced model interpretability.

## Technologies Used
Language:Python

Libraries:NumPy, Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn

Balancing:SMOTE

Visualization:Seaborn, Matplotlib

Model Deployment (Future):Streamlit/Flask


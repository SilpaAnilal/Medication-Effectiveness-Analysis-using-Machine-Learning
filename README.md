# Medication-Effectiveness-Analysis-using-Machine-Learning
Machine learning project for predicting 30-day hospital readmissions in diabetes patients, focusing on medication effectiveness and clinical data analysis.

# 📊 Project Overview
Hospital readmissions are a major concern in healthcare, particularly for chronic diseases like diabetes. This project aims to predict 30-day hospital readmissions in diabetes patients by analyzing various factors, especially the effectiveness of prescribed medications.By building multiple machine learning models, this study provides insights into patient care improvement and hospital resource management.

# Problem Statement
The primary objective is to develop a binary classification model that predicts whether a diabetic patient will be readmitted within 30 days after discharge, based on patient details, medical history, medications, and hospital encounters.

# 📂 Dataset
• Source: [Diabetic Data (UCI Machine Learning Repository)](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+Hospitals+for+Years+1999-2008)

• Records: 101,766 hospital admissions

• Features: 50+ attributes including demographics, diagnoses, medications, lab      results, etc.

# Workflow
**Data Preprocessing**
• Handle missing values
• Encode categorical variables
• Remove outliers
• Apply SMOTE for class balancing

**Exploratory Data Analysis (EDA)**
• Understand variable distributions
• Study medication and readmission correlations

**Model Building**
• Train and compare multiple machine learning models
• Perform hyperparameter tuning

**Evaluation**
• Use confusion matrix, ROC curve, accuracy, precision, recall, and F1 score
• Visualize feature importance

**Model Saving**
• Save the best model for deployment

## Results & Insights

• Best Model: XGBoost Classifier

• Achieved a balanced F1-Score, indicating effective prediction of minority class (readmitted patients).

• Stacking Ensemble improved generalization by combining multiple models.

• Medication types, number of prior admissions, and discharge disposition were among the most important features affecting readmission.

## Limitations

• Dataset may not reflect current clinical guidelines

• Imbalanced classes required heavy use of balancing techniques, which may affect real-world generalization.

• Lack of time-series data limits sequential pattern analysis(data is static).

• Medication adherence outside hospital visits is unknown.

## Future Work

•  Incorporate deep learning models (e.g., LSTM for sequential health data)

• Use real-time patient monitoring data

• Deploy the model using Flask or Streamlit for clinical decision support

• Integrate SHAP values for enhanced model interpretability.

## Technologies Used
**Language**: Python

**Libraries**: NumPy, Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn

**Balancing**: SMOTE

**Visualization**: Seaborn, Matplotlib

**Model Deployment (Future)**: Streamlit/Flask


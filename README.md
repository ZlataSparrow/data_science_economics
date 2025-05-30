# Heart Attack Risk Prediction (CDC BRFSS Data)

![Health Analytics](https://img.shields.io/badge/domain-health%20analytics-blue) ![Python](https://img.shields.io/badge/python-3.9%2B-green) ![License](https://img.shields.io/badge/license-MIT-orange)

A data science project predicting heart attack risk using CDC's Behavioral Risk Factor Surveillance System (BRFSS) survey data.

## 📌 Project Overview
- **Objective**: Predict individual heart attack risk based on 40+ health and demographic factors
- **Data Source**: [CDC BRFSS 2023 Dataset](https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system) via Kaggle
- **Key Features**:
  - Demographic indicators (age, gender, income)
  - Health metrics (BMI, cholesterol, blood pressure)
  - Lifestyle factors (smoking, exercise, alcohol consumption)

## 🗃️ Dataset Details
- **Full Name**: Behavioral Risk Factor Surveillance System (BRFSS)
- **Collection Method**: Annual telephone surveys by CDC
- **Versions Available**:
  - Raw dataset (contains missing values)
  - Cleaned dataset (NaNs removed/imputed)
- **Size**: ~400,000 respondents (2023 data)

## 🛠️ Technical Implementation
```python
# Sample code structure
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)  # Predicting heart attack risk


# Medical Insurance Cost Prediction 

Predicting healthcare insurance charges using advanced machine learning techniques.

This project builds a complete ML pipeline to model medical insurance costs based on demographic and lifestyle features. The workflow includes exploratory data analysis (EDA), preprocessing, multiple regression models, hyperparameter tuning, ensemble stacking, log-transform modeling, SHAP interpretability, and full residual diagnostics.

<img width="370" height="290" alt="image" src="https://github.com/user-attachments/assets/0766b9ee-5459-4d1a-90a9-f37058b5adbf" />
<img width="2989" height="989" alt="image" src="https://github.com/user-attachments/assets/e91697ce-237c-45bc-a9ce-f514bc82a390" />

<img width="355" height="226" alt="image" src="https://github.com/user-attachments/assets/b075ddb3-735f-4dc2-a28d-63e56407e421" />
<img width="350" height="224" alt="image" src="https://github.com/user-attachments/assets/6193af8c-256f-434a-b717-4c959f6df54f" />



<img width="468" height="400" alt="image" src="https://github.com/user-attachments/assets/a8ff325c-5f80-4e7b-932b-3efc19e1a709" />

<img width="468" height="400" alt="image" src="https://github.com/user-attachments/assets/ddff31e0-ba1a-4b23-9739-4bd50a8de659" />
<img width="405" height="255" alt="image" src="https://github.com/user-attachments/assets/5ccdebcf-bd43-45e2-a67c-25fb5b18852d" />
<img width="405" height="255" alt="image" src="https://github.com/user-attachments/assets/1b363054-a8e8-434e-b8e1-c6f86cd1f49d" />
<img width="409" height="270" alt="image" src="https://github.com/user-attachments/assets/5cb63fb8-2d82-4803-8265-9fdd16ea239c" />
<img width="447" height="270" alt="image" src="https://github.com/user-attachments/assets/7c12ef79-5761-4c24-828b-0f47b0d73afd" />

---

## Project Goal

To build an end-to-end regression system that predicts an individual’s medical insurance charges using demographic and health-related variables. The focus is on model accuracy, interpretability, and industry-standard ML workflow practices.

---

## Dataset Overview

Source: Medical Insurance Cost Dataset
Total rows: 1338

### Features

* age - Age of primary beneficiary
* sex - Gender (male/female)
* bmi - Body Mass Index
* children - Number of dependents
* smoker - Smoking status (yes/no)
* region - Residential region (NE, NW, SE, SW)
* charges - Target variable; total medical cost billed

### Common Uses

* Regression modeling
* Insurance pricing research
* Health economics
* ML education and feature engineering
* Understanding cost drivers in healthcare

---

## Methodology

### 1. Exploratory Data Analysis (EDA)

* Distribution plots of features
* Pairplots and correlation heatmaps
* Boxplots of charges across categories
* Statistical tests (t-tests, correlation metrics)

### 2. Data Preprocessing

* Train–test split
* StandardScaler for numeric features
* OneHotEncoder for categorical features
* ColumnTransformer pipeline
* Polynomial and interaction features

### 3. Models Implemented

The project compares a wide set of models:

Linear and Regularized Models

* Linear Regression
* Ridge, Lasso, ElasticNet
* Polynomial Regression

Tree-Based Models

* Random Forest
* GradientBoostingRegressor
* HistGradientBoostingRegressor
* XGBoost
* LightGBM
* CatBoost

Ensemble Techniques

* Gradient Boosting with hyperparameter tuning
* Log-transform regression
* Quantile regression (95th percentile)
* Stacking ensemble (CatBoost + LightGBM + GBR with Ridge meta-learner)

Explainability

* SHAP summary plots
* SHAP dependence analysis

Diagnostics

* Residual vs. prediction plots
* Residual distribution
* QQ plots
* Error analysis

---

## Final Model & Results

After evaluating all models, CatBoost delivered the strongest performance.

### Final Metrics

| Model                     | RMSE      |
| ------------------------- | --------- |
| Linear Regression         | ~6100     |
| Gradient Boosting (Tuned) | ~4620     |
| LightGBM                  | ~5002     |
| Stacking Model            | ~4434     |
| CatBoost (Best)       | ~4426 |

### Why CatBoost Won

* Handles categorical data natively
* Captures nonlinear interactions automatically
* Strong performance on small tabular datasets
* Excellent bias-variance balance
* Stable and fast training
* Highly interpretable with SHAP

---

## SHAP Insights (Feature Importance)

SHAP analysis revealed that:

* Smoker is by far the strongest predictor of insurance cost
* Age increases costs consistently
* BMI and BMI² capture obesity-related risk
* Age × BMI interactions matter
* Region has mild effect
* Sex and children have minimal impact

These patterns align with real-world medical cost dynamics.

---

## Residual Diagnostics

Residual analysis shows:

* Good performance on low–medium charges
* Increasing error variance for high-cost patients (heteroscedasticity)
* Some underprediction on extreme medical bills
* Residuals are right-skewed, consistent with healthcare claims data
* QQ plot confirms heavy-tailed distribution

This behavior is expected in insurance forecasting tasks.

---

## Future Improvements

Potential enhancements:

* Tweedie regression (industry standard for insurance)
* Optuna hyperparameter tuning
* CatBoost Bayesian optimization
* Conformal prediction intervals
* Partial Dependence Plots (PDP)
* Streamlit dashboard or FastAPI deployment

---


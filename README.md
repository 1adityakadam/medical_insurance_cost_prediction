# 🏥 Medical Insurance Cost Prediction
### A business-driven ML pipeline for healthcare cost modeling - built to support risk-based pricing, cost forecasting, and actuarial decision-making.

> *A business-driven machine learning project focused on driving measurable impact through scalable predictive modeling and AI-enhanced analytics.*

---

## 💼 Business Problem

Health insurers, self-insured employers, and benefits consultants face a costly challenge: **they cannot accurately price risk without understanding cost drivers at the individual level.**

When insurers misprice premiums - even by 10-15% - the financial consequences compound across thousands of policyholders. Overpricing drives customer churn. Underpricing creates unsustainable loss ratios. And without interpretable models, actuaries cannot explain *why* one individual costs 5× more than another.

**The question this project answers:**
> *Can we predict an individual's annual medical insurance charges with enough accuracy and interpretability to support premium pricing, risk segmentation, and cost forecasting decisions?*

---

## ❗ Why This Matters

Healthcare is a trillion-dollar industry where pricing decisions are made at scale.

- The average insured American incurs ~$8,000/year in medical costs - but the distribution is highly skewed: a small number of high-risk individuals drive the majority of claims
- Smokers incur, on average, **3-4× higher medical costs** than non-smokers - yet many pricing models fail to quantify *interaction effects* (e.g., smoking × BMI × age)
- Insurers and employers that rely on population averages instead of individual-level models leave significant money on the table - or worse, systemically undercharge the highest-risk segments

A model that improves RMSE by even $500/individual translates to **millions of dollars in premium accuracy** at portfolio scale.

---

## 🎯 Objective

Build an end-to-end regression system that:

1. Predicts individual medical insurance charges from demographic and lifestyle variables
2. Identifies and quantifies the key cost drivers through SHAP-based interpretability
3. Benchmarks multiple model families to find the best bias-variance tradeoff
4. Produces residual diagnostics that reveal where and why the model struggles
5. Delivers a framework transferable to real actuarial and health economics workflows

---

## 💰 Business Impact

*The following estimates are modeled on realistic insurance industry assumptions to illustrate how this work would translate in a production setting.*

| Impact Area | Estimate | Assumption |
|---|---|---|
| **Premium accuracy improvement** | ~$500 reduction in per-member RMSE vs. baseline | CatBoost RMSE ≈ 4,426 vs. Linear RMSE ≈ 6,100 |
| **Portfolio-level value** | ~$850K+ annual savings | 500-member employer plan × $500 RMSE delta × 34% cost-to-charges ratio |
| **High-risk identification** | Top 10% of members drive ~60-70% of costs | Quantile regression enables this segmentation directly |
| **Actuarial efficiency** | Reduced manual underwriting time | SHAP explanations provide instant, auditable factor attribution |
| **Strategic decisions enabled** | Risk-based benefit design, wellness program targeting, reinsurance thresholds | Smoker × BMI interaction identified as primary cost lever |

> **Note:** This is a portfolio project using the public Kaggle insurance dataset (1,338 records). Business impact figures are modeled assumptions to demonstrate how this framework would scale in production - not claims from a live deployment.

---

## 📊 Dataset Overview

**Source:** [Medical Insurance Cost Dataset - Kaggle](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset)

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Age of primary beneficiary |
| `sex` | Categorical | Gender (male/female) |
| `bmi` | Numeric | Body Mass Index |
| `children` | Numeric | Number of dependents covered |
| `smoker` | Categorical | Smoking status (yes/no) |
| `region` | Categorical | Residential region (NE, NW, SE, SW) |
| `charges` | **Target** | Total annual medical cost billed |

**1,338 rows · 7 features · 0 missing values**

The target variable (`charges`) is heavily right-skewed - a small cohort of high-cost patients (predominantly smokers with elevated BMI) drives the tail. This mirrors real-world insurance claims distributions.

---

## 🧠 Methodology & Decision Process

### Why not just use linear regression?

Linear regression was the natural baseline - interpretable, fast, and explainable to non-technical stakeholders. But a Welch's t-test confirmed smoker vs. non-smoker charges are statistically different at p < 0.0001, and pairplots revealed clearly nonlinear relationships between BMI, age, and charges. Linear models cannot capture the *multiplicative* interaction between smoking and BMI that dominates the cost distribution.

### Feature Engineering Choices

Rather than dropping polynomial expansion after seeing it improve linear models, I applied `PolynomialFeatures(degree=2)` to all numeric features - generating terms like `bmi²` and `age × bmi`. This was deliberate: BMI-squared captures the nonlinear obesity risk curve, and age × BMI reflects how metabolic risk compounds over time. These domain-informed features drove meaningful improvement even in the regularized models.

### Why CatBoost Over XGBoost or LightGBM?

Both are gradient boosting frameworks, but the key difference is **native categorical handling**. XGBoost and LightGBM require one-hot encoding for features like `smoker` and `region` - which can reduce the algorithm's ability to detect interaction effects across those splits. CatBoost's ordered target encoding preserves richer signal, especially on small datasets like this one where every split matters.

### Tradeoffs Considered

- **Log-transform of target:** Applied to reduce heteroscedasticity. Result: RMSE worsened slightly (4,662 vs. 4,426), suggesting CatBoost handles the skewed distribution natively better than a log-transform + simpler model.
- **Stacking ensemble:** Combining CatBoost + LightGBM + GBR with a Ridge meta-learner yielded RMSE ≈ 4,434 - marginally worse than CatBoost alone. This suggests the base models share correlated errors, limiting diversity. With more data, stacking would likely outperform.
- **Quantile regression (95th percentile):** Added specifically for high-cost patient identification - a use case more valuable to actuaries than mean prediction alone.

### Assumptions Made

- Mean charges used as baseline comparison (~$13,346)
- No external data enrichment (claims history, lab values, diagnoses) - real actuarial models would include these
- Train/test split: 80/20 with `random_state=42`; 10-fold cross-validation used to validate stability

---

## 📈 Model Results

| Model | RMSE | Notes |
|---|---|---|
| Linear Regression | ~6,100 | Baseline; fails on nonlinear interactions |
| Ridge / Lasso / ElasticNet | ~5,800-6,000 | Marginal improvement via regularization |
| Random Forest | ~5,200 | Better generalization; higher variance |
| XGBoost | ~4,900 | Solid; requires careful OHE preprocessing |
| LightGBM | ~5,002 | Fast; limited split gains on small dataset |
| Gradient Boosting (Tuned) | ~4,620 | GridSearchCV across 54 param combinations |
| Log-transformed GB | ~4,662 | Reduced heteroscedasticity; no RMSE gain |
| Stacking (CatBoost + LightGBM + GBR) | ~4,434 | Correlated errors limit ensemble uplift |
| **CatBoost (Final)** | **~4,426** | **Best: R² ≈ 0.84, MAE ≈ ~$3,000** |

---

## 🔍 SHAP: What's Actually Driving Costs?

SHAP analysis on the final model revealed a clear hierarchy of cost drivers:

**1. `smoker_yes` - dominant by a large margin.** Smokers' predictions shift upward by $10,000-$20,000+ compared to equivalent non-smokers. This matches clinical evidence: smoking significantly elevates cardiovascular, pulmonary, and oncological risk.

**2. `age` - consistent monotonic increase.** Every additional decade adds material charge risk, regardless of other factors.

**3. `bmi` and `bmi²` - captures the nonlinear obesity-risk curve.** The quadratic term confirms that risk *accelerates* above BMI 30, not just increases linearly.

**4. `age × bmi` interaction** - the combined effect of aging with elevated BMI compounds risk beyond additive effects alone.

**5. `region` - mild effect.** Regional healthcare pricing variation is real but secondary to lifestyle factors.

**6. `sex` and `children` - minimal predictive impact.** The t-test for sex (p ≈ 0.04) shows marginal difference, but effect size is small enough that the model correctly deprioritizes it.

> **Business translation:** A wellness program targeting smoking cessation + BMI reduction in the 40-55 age bracket would have the highest per-dollar ROI on claims reduction. SHAP makes this segmentation directly actionable.

---

## 🔬 Residual Diagnostics

Residual analysis revealed patterns consistent with real-world insurance claims data:

- **Low and mid-range charges (< $20K):** Model predictions are well-calibrated with small, symmetric residuals
- **High-cost patients (> $30K):** Systematic underprediction - the model pulls extreme values toward the mean (classic regression-to-mean behavior on small samples)
- **Heteroscedasticity confirmed:** Residual variance increases with predicted charge magnitude. This is expected in claims data and reflects the inherent unpredictability of catastrophic medical events - not a model failure
- **QQ plot:** Confirms heavy-tailed residuals, consistent with the Tweedie distribution commonly used in actuarial science

This is precisely why the quantile regression (95th percentile) component was included - to give actuaries a conservative upper-bound estimate for high-risk individuals, independent of the mean prediction.

---

## ⚙️ Tech Stack

**Data & Processing:** Python · Pandas · NumPy · Scikit-learn (ColumnTransformer, Pipelines, PolynomialFeatures, StandardScaler, OneHotEncoder) · SciPy

**Modeling:** Scikit-learn (LinearRegression, Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting, StackingRegressor) · XGBoost · LightGBM · CatBoost · GridSearchCV

**Interpretability & Diagnostics:** SHAP (summary plots, dependence plots) · Residual analysis · QQ plots

**Visualization:** Matplotlib · Seaborn

---

## 🤖 How AI Was Used

| Task | AI-Assisted? | My Role |
|---|---|---|
| Pipeline architecture | Partially (ChatGPT for ColumnTransformer syntax) | I decided what features to encode, why, and how |
| SHAP interpretation write-up | No | Written entirely from my own analysis of the plots |
| Model selection rationale | No | My decision based on dataset characteristics |
| README business framing | ChatGPT suggested structure | I wrote and validated all content and numbers |
| Hyperparameter grid design | Manual | I defined the search space from model documentation |
| Code debugging | ChatGPT for StackingRegressor import error | I validated the fix and understood why it worked |

**Principle:** AI accelerated execution. Every analytical decision - what to model, why it matters, what the results mean for a business - was mine.

---

## 🌍 How This Framework Applies Elsewhere

This is not a one-off insurance project. The methodology is transferable across any domain where the target is continuous, right-skewed, and driven by a small number of high-impact features - and where interpretability is required alongside accuracy.

- **Employee benefits consulting:** Predict per-employee healthcare spend for self-insured employers → inform benefit design decisions
- **Hospital revenue forecasting:** Predict patient billing from demographic + clinical inputs → improve CFO-level financial planning
- **Life & disability insurance:** Apply the same SHAP-driven feature attribution to mortality and disability risk models
- **Retail / e-commerce:** Replace `charges` with `customer lifetime value`; replace `smoker` with `loyalty status` - same pipeline, same logic
- **HR / workforce analytics:** Predict employee absenteeism cost; SHAP reveals which interventions have the highest ROI

The core framework - EDA → statistical validation → pipeline preprocessing → model benchmarking → SHAP interpretability → residual diagnostics - is industry-agnostic.

---

## 📋 Step-by-Step Reproduction Guide

**1. Data Acquisition**
Download `insurance.csv` from [Kaggle](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset).

**2. Environment Setup**
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost shap matplotlib seaborn scipy
```

**3. EDA**
Run `sns.pairplot(df, diag_kind='kde')`, boxplots of charges by categorical feature, correlation heatmap on numeric features, and Welch's t-test to statistically validate the smoker/non-smoker charge difference.

**4. Preprocessing Pipeline**
```python
numerical_transformer = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])
categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])
```

**5. Model Training**
Train each model family (linear → tree → boosted → ensemble) using a consistent Pipeline wrapper. Use `KFold(n_splits=10)` with `cross_val_score` for comparable CV-RMSE across all models.

**6. Hyperparameter Tuning**
Apply `GridSearchCV` with 10-fold CV on GradientBoosting. Key search parameters: `n_estimators`, `learning_rate`, `max_depth`, `subsample`.

**7. CatBoost Final Model**
```python
cat_model = CatBoostRegressor(
    depth=6, learning_rate=0.05, n_estimators=1000,
    loss_function='RMSE', random_seed=42, verbose=False
)
cat_model.fit(X_train, y_train.values.reshape(-1),
              cat_features=[X.columns.get_loc(c) for c in cat_cols])
```

**8. SHAP Interpretability**
Wrap the stacked predictor in a function, extract transformed feature names from the preprocessor, and run `shap.summary_plot()` with `shap.dependence_plot()` for top features.

**9. Residual Diagnostics**
Plot: residuals vs. predicted values, residual distribution histogram, QQ plot. Expect right-skewed residuals and increasing variance at high charges.

**10. Business Interpretation**
Translate SHAP outputs into plain stakeholder language: *"Smoking status alone shifts predicted annual charges upward by $10,000-$20,000 - this single variable justifies targeted cessation programs purely from a cost-avoidance standpoint."*

---

## 📖 Context

**Project type:** Independent portfolio project  
**Role:** Solo end-to-end - data exploration, feature engineering, modeling, interpretability, documentation  
**Stakeholder simulation:** Designed as if presenting to an actuarial team and a non-technical benefits director simultaneously - requiring both technical rigor and plain-language business translation  
**Constraints:** Public dataset (1,338 rows); no claims history, diagnosis codes, or lab values that a real insurer would have. Results are directional, not production-ready.

---

## 💡 Key Learnings

**What I would improve:**
- Implement **Tweedie regression** (the actuarial industry standard for skewed, zero-inflated cost distributions) as the primary benchmark from the start, not just a future roadmap item
- Use **Optuna** instead of GridSearchCV - significantly more efficient on larger search spaces and better suited for CatBoost's hyperparameter landscape
- Add **conformal prediction intervals** to produce calibrated uncertainty bounds around each prediction - critical for any real pricing application where regulatory defensibility matters

**What surprised me:**
- Log-transforming the target actually *worsened* RMSE slightly despite visually reducing heteroscedasticity. CatBoost handles the skewed distribution natively better than expected
- LightGBM's "No further splits with positive gain" warnings reflected something real: on small datasets with strong nonlinear interactions, leaf-wise splitting can underperform depth-wise methods like CatBoost
- SHAP confirmed domain knowledge almost exactly - smoking dominates, age and BMI compound, region barely registers. When ML output aligns with clinical reality, that's a signal the model is capturing meaningful signal, not noise

**Business insight gained:**
The model's underprediction of extreme high-cost cases is not a technical failure - it's a fundamental property of rare, catastrophic medical events. For insurers, this is precisely why stop-loss reinsurance exists. A model like this should inform *expected* costs; a separate risk layer handles the tail.

---

## 🚀 Future Roadmap

- [ ] Tweedie regression for actuarially-aligned loss function
- [ ] Optuna hyperparameter optimization (replace GridSearchCV)
- [ ] Conformal prediction intervals for calibrated uncertainty
- [ ] Partial Dependence Plots (PDP) for the smoker × BMI interaction surface
- [ ] Streamlit dashboard: input demographics → get predicted cost + SHAP breakdown in real time
- [ ] FastAPI deployment for integration into a benefits cost calculator

---

## 🤝 Let's Connect

If you're working on health economics modeling, insurance analytics, or any domain where interpretability and business impact need to coexist - I'd welcome the conversation.

Feedback on methodology, business framing, or model choices is genuinely valuable to me. Connect on [LinkedIn](#) or open an issue on this repo.

---

*Built with Python · Scikit-learn · CatBoost · SHAP · Real-world actuarial thinking*

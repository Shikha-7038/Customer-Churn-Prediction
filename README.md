# Customer Churn Prediction Model

A production-ready machine learning system that predicts customer churn, explains key drivers, and triggers retention actions.

## 🎯 Project Overview

This project builds a complete churn prediction pipeline that:
- **Predicts** which customers are likely to churn in the next billing cycle
- **Explains** the key factors driving churn for each customer
- **Recommends** specific retention actions based on risk profile
- **Serves** predictions via a REST API (FastAPI)
- **Generates** actionable business insights

## 📊 Business Impact

- **Reduce Customer Loss**: Identify at-risk customers before they leave
- **Increase Revenue**: Target retention offers effectively
- **Lower CAC**: Save on customer acquisition costs
- **Improve LTV**: Extend customer lifetime value

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Data Processing | Pandas, NumPy |
| ML Models | XGBoost, Random Forest, Logistic Regression |
| Optimization | Optuna |
| Explainability | SHAP |
| API | FastAPI, Uvicorn |
| Visualization | Matplotlib, Seaborn |
| Model Management | Joblib |

## 📁 Project Structure
```
Customer-Churn-Prediction/
├── data/ # Dataset storage
├── notebooks/ # Jupyter notebooks
├── src/ # Source code
│ ├── generate_data.py # Synthetic data generation
│ ├── features.py # Feature engineering
│ ├── pipeline.py # Preprocessing pipeline
│ ├── eda.py # Exploratory analysis
│ ├── train_model.py # Model training
│ └── shap_explainability.py # SHAP analysis
├── serving/ # API service
│ └── app.py # FastAPI application
├── models/ # Saved models
├── outputs/ # Results and predictions
├── images/ # Visualizations
├── main.py # Main execution script
├── requirements.txt # Dependencies
└── Dockerfile # Container configuration
```

## 🚀 Quick Start

### 1. Clone the Repository

git clone https://github.com/yourusername/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

2. Set Up Virtual Environment
bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate

3. Install Dependencies
bash
pip install -r requirements.txt

4. Run Complete Pipeline
bash
python main.py

This will:
Generate synthetic customer data
Perform EDA and create visualizations
Train multiple models (Logistic Regression, Random Forest, XGBoost)
Optimize XGBoost with Optuna
Generate SHAP explanations
Save models and predictions

5. Start API Server

python -m uvicorn serving.app:app --reload
Or run after main.py completes and select 'y' to start the server.

📡 API Endpoints
POST /score
Predict churn for a single customer

json
{
  "customer_id": "CUST_001",
  "billing_amount": 49.99,
  "last_payment_days_ago": 5,
  "plan_tier": "Standard",
  "tenure_months": 12,
  "monthly_usage_hours": 45,
  "active_days": 22,
  "login_count": 30,
  "avg_session_min": 25,
  "device_count": 2,
  "add_on_count": 1,
  "support_tickets": 0,
  "sla_breaches": 0,
  "promotions_redeemed": 1,
  "email_opens": 5,
  "email_clicks": 2,
  "last_campaign_days_ago": 10,
  "nps_score": 8,
  "region": "North",
  "is_autopay": true,
  "is_discounted": false,
  "has_family_bundle": false
}
POST /batch_score
Predict churn for multiple customers

POST /explain
Get explanation and recommendations for a prediction

GET /health
Check API health status

📊 Results
Model Performance
Model	PR-AUC	ROC-AUC	F1-Score
Logistic Regression	0.52	0.78	0.45
Random Forest	0.58	0.82	0.51
XGBoost	0.65	0.86	0.58
Calibrated XGBoost	0.64	0.86	0.58
Key Drivers of Churn
Low Engagement: Customers with <15 active days have 3x higher churn

Support Issues: Each support ticket increases churn by ~8%

Price Sensitivity: High billing + low usage = high churn risk

New Customers: First 3 months have highest churn rate

Payment Problems: Late payments strongly indicate churn

Actionable Segments
Risk Level	Criteria	Recommended Action
High Risk	>50% churn probability	Immediate retention offer + support escalation
Medium Risk	25-50% churn probability	Targeted campaign + satisfaction survey
Low Risk	<25% churn probability	Maintain engagement + upsell opportunities

📈 Visualization Examples
The pipeline generates the following visualizations in the images/ folder:

churn_distribution.png - Overall churn rate

churn_by_plan.png - Churn rate by plan tier

feature_distributions.png - Feature distributions by churn status

correlation_matrix.png - Feature correlations

shap_global_importance.png - SHAP feature importance

top_features.png - Top 10 most important features
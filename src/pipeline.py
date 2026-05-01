"""
Preprocessing Pipeline for Churn Prediction
Handles missing values, scaling, and encoding
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def create_preprocessor():
    """
    Create a column transformer for preprocessing
    
    Numerical features: median imputation + standardization
    Categorical features: mode imputation + one-hot encoding
    """
    
    # Numerical features (continuous)
    NUM_FEATURES = [
        'billing_amount', 'last_payment_days_ago', 'tenure_months',
        'monthly_usage_hours', 'active_days', 'login_count', 'avg_session_min',
        'device_count', 'add_on_count', 'support_tickets', 'sla_breaches',
        'promotions_redeemed', 'email_opens', 'email_clicks', 
        'last_campaign_days_ago', 'nps_score',
        # Engineered features
        'engagement_rate', 'usage_per_login', 'support_intensity', 'email_ctr',
        'price_to_tenure', 'recency_score', 'loyalty_score', 'value_at_risk',
        'inactivity_ratio'
    ]
    
    # Categorical features
    CAT_FEATURES = [
        'plan_tier', 'region', 'is_autopay', 'is_discounted', 'has_family_bundle',
        'upsell_potential', 'downgrade_risk'
    ]
    
    # Numerical pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Column transformer
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, NUM_FEATURES),
        ('cat', cat_pipeline, CAT_FEATURES)
    ])
    
    return preprocessor, NUM_FEATURES, CAT_FEATURES

# For backward compatibility
def create_preprocessing_pipeline():
    preprocessor, _, _ = create_preprocessor()
    return preprocessor

if __name__ == "__main__":
    preprocessor, num_feat, cat_feat = create_preprocessor()
    
    print("Created preprocessor with:")
    print(f"  - {len(num_feat)} numerical features")
    print(f"  - {len(cat_feat)} categorical features")
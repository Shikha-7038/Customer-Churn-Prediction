"""
Feature Engineering for Churn Prediction
Creates business-relevant features from raw data
"""

import pandas as pd
import numpy as np
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def add_features(df):
    """
    Add engineered features for better churn prediction
    
    Features created:
    1. engagement_rate - percentage of days active
    2. usage_per_login - efficiency of usage
    3. support_intensity - weighted support interactions
    4. email_ctr - click-through rate on emails
    5. price_to_tenure - price sensitivity relative to tenure
    6. recency_score - how recent was last activity
    7. loyalty_score - combination of tenure and engagement
    """
    
    df = df.copy()
    
    # Engagement rate (active days / 30)
    df['engagement_rate'] = (df['active_days'] / 30.0).clip(0, 1)
    
    # Usage per login (efficiency)
    df['usage_per_login'] = (df['monthly_usage_hours'] / (df['login_count'] + 1e-3))
    df['usage_per_login'] = df['usage_per_login'].clip(0, 100)
    
    # Support intensity (tickets + 3x SLA breaches)
    df['support_intensity'] = df['support_tickets'] + 3 * df['sla_breaches']
    
    # Email CTR (clicks / opens)
    df['email_ctr'] = (df['email_clicks'] / (df['email_opens'] + 1e-3))
    df['email_ctr'] = df['email_ctr'].clip(0, 1)
    
    # Price to tenure ratio (how much they pay relative to loyalty)
    df['price_to_tenure'] = df['billing_amount'] / (df['tenure_months'] + 1e-3)
    
    # Recency score (inverse of days since payment)
    df['recency_score'] = np.exp(-df['last_payment_days_ago'] / 30)
    
    # Loyalty score (high tenure + high engagement)
    df['loyalty_score'] = (df['tenure_months'] / 60) * df['engagement_rate']
    
    # Value at risk (billing amount * churn risk proxy)
    df['value_at_risk'] = df['billing_amount'] * (1 - df['engagement_rate'])
    
    # Inactivity ratio
    df['inactivity_ratio'] = (30 - df['active_days']) / 30
    
    # Upsell potential (if high usage on basic plan)
    df['upsell_potential'] = ((df['monthly_usage_hours'] > 50) & 
                               (df['plan_tier'] == 'Basic')).astype(int)
    
    # Downgrade risk (low usage on premium plan)
    df['downgrade_risk'] = ((df['monthly_usage_hours'] < 20) & 
                             (df['plan_tier'] == 'Premium')).astype(int)
    
    return df

if __name__ == "__main__":
    # Test the feature engineering
    parquet_path = os.path.join(DATA_DIR, 'churn_frame.parquet')
    
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        df_with_features = add_features(df)
        
        print("Original columns:", len(df.columns))
        print("Columns after feature engineering:", len(df_with_features.columns))
        print("\nNew features created:")
        new_features = ['engagement_rate', 'usage_per_login', 'support_intensity', 
                        'email_ctr', 'price_to_tenure', 'recency_score', 
                        'loyalty_score', 'value_at_risk', 'inactivity_ratio',
                        'upsell_potential', 'downgrade_risk']
        for feat in new_features:
            print(f"  - {feat}: mean={df_with_features[feat].mean():.3f}")
    else:
        print(f"Data file not found at {parquet_path}")
        print("Please run generate_data.py first")
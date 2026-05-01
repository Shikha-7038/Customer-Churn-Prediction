"""
Synthetic Customer Churn Data Generator
Simulates real-world customer behavior for churn prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

# Get the project root directory (parent of src folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def generate_churn_data(n_customers=10000):
    """
    Generate synthetic customer data with churn patterns
    
    Business logic simulated:
    - Customers with low engagement are more likely to churn
    - High support tickets indicate dissatisfaction
    - Price sensitivity affects churn
    - Tenure affects loyalty
    """
    
    data = []
    
    for i in range(n_customers):
        customer_id = f"CUST_{i+1:06d}"
        
        # Tenure (months) - longer tenure = lower churn probability
        tenure_months = np.random.exponential(scale=18)
        tenure_months = min(max(tenure_months, 1), 60)
        
        # Plan tier: Basic, Standard, Premium
        plan_tier = np.random.choice(['Basic', 'Standard', 'Premium'], 
                                      p=[0.4, 0.4, 0.2])
        
        # Monthly billing amount based on plan
        if plan_tier == 'Basic':
            base_billing = np.random.uniform(20, 40)
        elif plan_tier == 'Standard':
            base_billing = np.random.uniform(40, 70)
        else:
            base_billing = np.random.uniform(70, 120)
        
        billing_amount = base_billing * (1 + np.random.uniform(-0.1, 0.1))
        
        # Engagement metrics
        monthly_usage_hours = np.random.gamma(shape=2, scale=15)
        monthly_usage_hours = min(monthly_usage_hours, 200)
        
        active_days = np.random.binomial(30, 0.6)
        
        login_count = int(np.random.exponential(scale=20))
        login_count = min(login_count, 200)
        
        avg_session_min = np.random.normal(25, 10)
        avg_session_min = max(avg_session_min, 1)
        
        # Support interactions (higher = more likely to churn)
        support_tickets = np.random.poisson(0.5)
        sla_breaches = np.random.binomial(1, 0.05) * np.random.poisson(0.3)
        
        # Payment behavior
        last_payment_days_ago = np.random.exponential(scale=15)
        last_payment_days_ago = min(last_payment_days_ago, 60)
        
        is_autopay = np.random.choice([True, False], p=[0.7, 0.3])
        
        # Marketing engagement
        email_opens = np.random.poisson(3)
        email_clicks = np.random.binomial(email_opens, 0.2)
        
        promotions_redeemed = np.random.poisson(0.5)
        
        # NPS score (0-10, lower = higher churn)
        nps_score = np.random.normal(7, 2)
        nps_score = min(max(nps_score, 0), 10)
        
        # Region
        region = np.random.choice(['North', 'South', 'East', 'West'], 
                                   p=[0.25, 0.25, 0.25, 0.25])
        
        # Features
        device_count = np.random.poisson(2) + 1
        add_on_count = np.random.poisson(1)
        has_family_bundle = np.random.choice([True, False], p=[0.3, 0.7])
        is_discounted = np.random.choice([True, False], p=[0.2, 0.8])
        
        # CHURN LOGIC (simulating real-world patterns)
        churn_prob = 0.05  # base churn rate
        
        # Tenure effect: new customers more likely to churn
        if tenure_months < 3:
            churn_prob += 0.15
        elif tenure_months < 6:
            churn_prob += 0.08
        elif tenure_months > 24:
            churn_prob -= 0.05
        
        # Engagement effect
        if monthly_usage_hours < 20:
            churn_prob += 0.12
        if active_days < 15:
            churn_prob += 0.10
        if login_count < 10:
            churn_prob += 0.08
        
        # Support effect
        churn_prob += support_tickets * 0.08
        churn_prob += sla_breaches * 0.10
        
        # Payment effect
        if last_payment_days_ago > 30:
            churn_prob += 0.15
        if not is_autopay:
            churn_prob += 0.05
        
        # Plan and price effect
        if billing_amount > 80 and monthly_usage_hours < 30:
            churn_prob += 0.10
        
        # NPS effect
        if nps_score < 6:
            churn_prob += 0.12
        elif nps_score > 8:
            churn_prob -= 0.05
        
        # Random noise
        churn_prob += np.random.normal(0, 0.03)
        
        # Cap the probability
        churn_prob = min(max(churn_prob, 0), 0.95)
        
        # Determine churn
        churned = np.random.random() < churn_prob
        
        data.append({
            'customer_id': customer_id,
            'cycle_start': '2024-01-01',
            'cycle_end': '2024-01-31',
            'billing_amount': round(billing_amount, 2),
            'last_payment_days_ago': round(last_payment_days_ago, 1),
            'plan_tier': plan_tier,
            'tenure_months': round(tenure_months, 1),
            'monthly_usage_hours': round(monthly_usage_hours, 1),
            'active_days': active_days,
            'login_count': login_count,
            'avg_session_min': round(avg_session_min, 1),
            'device_count': device_count,
            'add_on_count': add_on_count,
            'support_tickets': support_tickets,
            'sla_breaches': round(sla_breaches, 1),
            'promotions_redeemed': promotions_redeemed,
            'email_opens': email_opens,
            'email_clicks': email_clicks,
            'last_campaign_days_ago': round(np.random.exponential(10), 1),
            'nps_score': round(nps_score, 1),
            'region': region,
            'is_autopay': is_autopay,
            'is_discounted': is_discounted,
            'has_family_bundle': has_family_bundle,
            'churned_next_cycle': int(churned)
        })
    
    df = pd.DataFrame(data)
    
    # Add date columns
    df['cycle_start'] = pd.to_datetime(df['cycle_start'])
    df['cycle_end'] = pd.to_datetime(df['cycle_end'])
    
    return df

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print("Generating synthetic customer data...")
    
    df = generate_churn_data(10000)
    
    # Save to CSV (always works)
    csv_path = os.path.join(DATA_DIR, 'churn_frame.csv')
    df.to_csv(csv_path, index=False)
    print(f"[OK] CSV saved to: {csv_path}")
    
    # Try to save as Parquet (optional, requires pyarrow)
    parquet_path = os.path.join(DATA_DIR, 'churn_frame.parquet')
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"[OK] Parquet saved to: {parquet_path}")
    except ImportError:
        print("[WARN] Pyarrow not installed, skipping Parquet format")
        print("       To install: pip install pyarrow")
    except Exception as e:
        print(f"[WARN] Could not save Parquet: {e}")
    
    print(f"\n[SUCCESS] Generated {len(df)} customer records")
    print(f"[INFO] Churn rate: {df['churned_next_cycle'].mean():.2%}")
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
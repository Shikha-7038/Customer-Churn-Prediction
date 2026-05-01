"""
Exploratory Data Analysis for Churn Prediction
Visualizes patterns and insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import add_features

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')

def perform_eda():
    """Perform comprehensive EDA on churn data"""
    
    # Create directories
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # Load data - try CSV first, then Parquet
    print("Loading data...")
    csv_path = os.path.join(DATA_DIR, 'churn_frame.csv')
    parquet_path = os.path.join(DATA_DIR, 'churn_frame.parquet')
    
    df = None
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV from: {csv_path}")
    elif os.path.exists(parquet_path):
        try:
            df = pd.read_parquet(parquet_path)
            print(f"Loaded Parquet from: {parquet_path}")
        except:
            print("Parquet read failed, trying CSV...")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
    
    if df is None:
        print(f"[ERROR] No data file found. Please run generate_data.py first")
        return None
    
    # Convert date columns if they exist
    if 'cycle_start' in df.columns:
        df['cycle_start'] = pd.to_datetime(df['cycle_start'])
        df['cycle_end'] = pd.to_datetime(df['cycle_end'])
    
    df = add_features(df)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Churn rate: {df['churned_next_cycle'].mean():.2%}")
    
    # 1. Churn Distribution
    plt.figure(figsize=(8, 6))
    churn_counts = df['churned_next_cycle'].value_counts()
    plt.pie(churn_counts, labels=['Not Churned', 'Churned'], 
            autopct='%1.1f%%', startangle=90, explode=(0, 0.05))
    plt.title('Customer Churn Distribution', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(IMAGES_DIR, 'churn_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Churn distribution chart saved")
    
    # 2. Churn by Plan Tier
    plt.figure(figsize=(10, 6))
    plan_churn = df.groupby('plan_tier')['churned_next_cycle'].agg(['mean', 'count'])
    plan_churn = plan_churn.sort_values('mean', ascending=False)
    bars = plt.bar(plan_churn.index, plan_churn['mean'] * 100, 
                   color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    plt.ylabel('Churn Rate (%)', fontsize=12)
    plt.xlabel('Plan Tier', fontsize=12)
    plt.title('Churn Rate by Plan Type', fontsize=14, fontweight='bold')
    
    for bar, rate in zip(bars, plan_churn['mean'] * 100):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'churn_by_plan.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Churn by plan chart saved")
    
    # 3. Key Feature Distributions by Churn
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    features = ['tenure_months', 'monthly_usage_hours', 'support_tickets',
                'nps_score', 'engagement_rate', 'billing_amount']
    
    for idx, feature in enumerate(features):
        row, col = idx // 3, idx % 3
        for churn_val, color, label in zip([0, 1], ['#4ecdc4', '#ff6b6b'], 
                                           ['Not Churned', 'Churned']):
            axes[row, col].hist(df[df['churned_next_cycle'] == churn_val][feature], 
                               bins=30, alpha=0.6, color=color, label=label, density=True)
        axes[row, col].set_xlabel(feature.replace('_', ' ').title(), fontsize=10)
        axes[row, col].set_ylabel('Density', fontsize=10)
        axes[row, col].legend()
        axes[row, col].set_title(f'{feature.replace("_", " ").title()} Distribution', fontsize=11)
    
    plt.suptitle('Feature Distributions by Churn Status', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'feature_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Feature distributions saved")
    
    # 4. Correlation Matrix
    plt.figure(figsize=(14, 12))
    numeric_cols = ['tenure_months', 'monthly_usage_hours', 'active_days', 
                    'login_count', 'support_tickets', 'nps_score', 
                    'billing_amount', 'engagement_rate', 'loyalty_score',
                    'churned_next_cycle']
    # Only use columns that exist
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    corr_matrix = df[numeric_cols].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'correlation_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Correlation matrix saved")
    
    # 5. Churn by Engagement Level
    plt.figure(figsize=(10, 6))
    df['engagement_level'] = pd.cut(df['engagement_rate'], 
                                     bins=[0, 0.3, 0.6, 1.0], 
                                     labels=['Low', 'Medium', 'High'])
    engagement_churn = df.groupby('engagement_level', observed=True)['churned_next_cycle'].mean()
    
    colors = ['#ff6b6b', '#feca57', '#4ecdc4']
    bars = plt.bar(engagement_churn.index, engagement_churn.values * 100, color=colors)
    plt.ylabel('Churn Rate (%)', fontsize=12)
    plt.xlabel('Engagement Level', fontsize=12)
    plt.title('Churn Rate by Customer Engagement Level', fontsize=14, fontweight='bold')
    
    for bar, rate in zip(bars, engagement_churn.values * 100):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, 'churn_by_engagement.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Engagement vs churn chart saved")
    
    # Generate summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    print(f"\nOverall Statistics:")
    print(f"  Total Customers: {len(df):,}")
    print(f"  Churned Customers: {df['churned_next_cycle'].sum():,}")
    print(f"  Churn Rate: {df['churned_next_cycle'].mean():.2%}")
    
    print(f"\nChurn by Plan Tier:")
    for plan in df['plan_tier'].unique():
        plan_df = df[df['plan_tier'] == plan]
        print(f"  {plan}: {plan_df['churned_next_cycle'].mean():.2%} ({len(plan_df):,} customers)")
    
    print(f"\nChurn by Region:")
    for region in df['region'].unique():
        region_df = df[df['region'] == region]
        print(f"  {region}: {region_df['churned_next_cycle'].mean():.2%}")
    
    print(f"\nKey Feature Averages (Churned vs Not Churned):")
    compare_features = ['tenure_months', 'monthly_usage_hours', 'engagement_rate', 
                        'nps_score', 'support_tickets', 'loyalty_score']
    for feat in compare_features:
        if feat in df.columns:
            churned_mean = df[df['churned_next_cycle'] == 1][feat].mean()
            not_churned_mean = df[df['churned_next_cycle'] == 0][feat].mean()
            print(f"  {feat}: Churned={churned_mean:.2f} | Not Churned={not_churned_mean:.2f}")
    
    return df

if __name__ == "__main__":
    df = perform_eda()
    if df is not None:
        print("\n[SUCCESS] EDA completed successfully! Check the 'images' folder for visualizations.")
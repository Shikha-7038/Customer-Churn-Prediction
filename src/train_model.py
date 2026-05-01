"""
Model Training for Churn Prediction
Trains multiple models and selects the best one
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, average_precision_score,
                            classification_report, confusion_matrix)
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import add_features
from src.pipeline import create_preprocessor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')

def load_and_prepare_data():
    """Load data and prepare for modeling"""
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
            df = pd.read_csv(csv_path)
            print(f"Loaded CSV from: {csv_path}")
    
    if df is None:
        raise FileNotFoundError(f"No data file found. Please run generate_data.py first")
    
    df = add_features(df)
    
    # Get preprocessor and feature lists
    preprocessor, num_feat, cat_feat = create_preprocessor()
    feature_cols = num_feat + cat_feat
    
    # Ensure all features exist in dataframe
    available_features = [f for f in feature_cols if f in df.columns]
    missing_features = set(feature_cols) - set(available_features)
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        feature_cols = available_features
    
    X = df[feature_cols]
    y = df['churned_next_cycle']
    
    # Split data
    split_idx = int(len(df) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_baseline_models(X_train, X_test, y_train, y_test, preprocessor):
    """Train and evaluate baseline models"""
    print("\n" + "="*60)
    print("TRAINING BASELINE MODELS")
    print("="*60)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, 
            class_weight='balanced',
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        try:
            # Transform features
            X_train_trans = preprocessor.fit_transform(X_train)
            X_test_trans = preprocessor.transform(X_test)
            
            # Train model
            model.fit(X_train_trans, y_train)
            
            # Predict
            y_pred = model.predict(X_test_trans)
            y_pred_proba = model.predict_proba(X_test_trans)[:, 1]
            
            # Evaluate
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'pr_auc': average_precision_score(y_test, y_pred_proba)
            }
            
            print(f"  Accuracy: {results[name]['accuracy']:.4f}")
            print(f"  Precision: {results[name]['precision']:.4f}")
            print(f"  Recall: {results[name]['recall']:.4f}")
            print(f"  F1-Score: {results[name]['f1']:.4f}")
            print(f"  ROC-AUC: {results[name]['roc_auc']:.4f}")
            print(f"  PR-AUC: {results[name]['pr_auc']:.4f}")
            
            # Save model
            model_path = os.path.join(MODELS_DIR, f'{name.lower().replace(" ", "_")}.joblib')
            joblib.dump(model, model_path)
            print(f"  Saved to: {model_path}")
            
        except Exception as e:
            print(f"  Error training {name}: {e}")
            results[name] = {
                'accuracy': 0, 'precision': 0, 'recall': 0, 
                'f1': 0, 'roc_auc': 0, 'pr_auc': 0
            }
    
    return results

def train_xgboost_model(X_train, X_test, y_train, y_test, preprocessor):
    """Train XGBoost model (simplified, no Optuna for faster execution)"""
    print("\n" + "="*60)
    print("TRAINING XGBOOST MODEL")
    print("="*60)
    
    try:
        import xgboost as xgb
        
        # Transform features
        X_train_trans = preprocessor.fit_transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
        
        # Calculate scale_pos_weight for imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        model.fit(X_train_trans, y_train)
        
        # Calibrate
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
        calibrated_model.fit(X_train_trans, y_train)
        
        # Evaluate
        y_pred_proba = calibrated_model.predict_proba(X_test_trans)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba)
        }
        
        print("\nXGBoost Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return model, calibrated_model, metrics
        
    except ImportError:
        print("XGBoost not installed. Skipping...")
        return None, None, None

def main():
    """Main training pipeline"""
    print("="*60)
    print("CUSTOMER CHURN PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # Load data
    try:
        X_train, X_test, y_train, y_test, preprocessor = load_and_prepare_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"\nData split:")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Testing samples: {len(X_test):,}")
    print(f"  Churn rate (train): {y_train.mean():.2%}")
    print(f"  Churn rate (test): {y_test.mean():.2%}")
    
    # Train baseline models
    baseline_results = train_baseline_models(X_train, X_test, y_train, y_test, preprocessor)
    
    # Train XGBoost
    xgb_model, calibrated_xgb, xgb_metrics = train_xgboost_model(X_train, X_test, y_train, y_test, preprocessor)
    
    # Save the best model
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    # Save preprocessor
    preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.joblib')
    joblib.dump(preprocessor, preprocessor_path)
    print(f"[OK] Preprocessor saved to {preprocessor_path}")
    
    # Save calibrated model if available
    if calibrated_xgb is not None:
        model_path = os.path.join(MODELS_DIR, 'churn_model.joblib')
        joblib.dump(calibrated_xgb, model_path)
        print(f"[OK] Calibrated model saved to {model_path}")
    else:
        # Save Random Forest as fallback
        rf_path = os.path.join(MODELS_DIR, 'churn_model.joblib')
        rf_model = joblib.load(os.path.join(MODELS_DIR, 'random_forest.joblib'))
        joblib.dump(rf_model, rf_path)
        print(f"[OK] Random Forest model saved as fallback to {rf_path}")
    
    # Compare models
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    comparison_data = []
    for name, metrics in baseline_results.items():
        comparison_data.append({
            'Model': name,
            'PR-AUC': metrics['pr_auc'],
            'ROC-AUC': metrics['roc_auc'],
            'F1-Score': metrics['f1']
        })
    
    if xgb_metrics is not None:
        comparison_data.append({
            'Model': 'XGBoost',
            'PR-AUC': xgb_metrics['pr_auc'],
            'ROC-AUC': xgb_metrics['roc_auc'],
            'F1-Score': xgb_metrics['f1']
        })
    
    comparison = pd.DataFrame(comparison_data)
    print(comparison.to_string(index=False))
    
    # Save comparison
    comparison_path = os.path.join(OUTPUTS_DIR, 'model_comparison.csv')
    comparison.to_csv(comparison_path, index=False)
    print(f"\n[OK] Model comparison saved to {comparison_path}")
    
    print("\n[SUCCESS] Model training completed successfully!")

if __name__ == "__main__":
    main()
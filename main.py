"""
Main Execution Script for Customer Churn Prediction Project
Run this file to execute the complete pipeline
"""

import subprocess
import sys
import os

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print_header(description)
    
    # Get the full path
    full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_path)
    
    if not os.path.exists(full_path):
        print(f"[ERROR] Script not found: {full_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, full_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        print(result.stdout)
        if result.stderr:
            print("WARNINGS/ERRORS:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"[ERROR] Running {script_path}: {e}")
        return False

def main():
    """Run the complete project pipeline"""
    
    print_header("CUSTOMER CHURN PREDICTION PROJECT")
    print("This script will run the complete pipeline:")
    print("1. Generate synthetic data")
    print("2. Perform exploratory data analysis")
    print("3. Train and evaluate models")
    print("4. Generate SHAP explanations")
    print("5. Start FastAPI server (optional)")
    
    # Create necessary directories
    for dir_name in ["data", "models", "outputs", "images", "notebooks"]:
        os.makedirs(dir_name, exist_ok=True)
    
    # Step 1: Generate data
    if not run_script("src/generate_data.py", "STEP 1: GENERATING SYNTHETIC DATA"):
        print("[ERROR] Data generation failed!")
        return
    
    # Step 2: EDA
    if not run_script("src/eda.py", "STEP 2: EXPLORATORY DATA ANALYSIS"):
        print("[WARN] EDA had issues, continuing with model training...")
    
    # Step 3: Train models
    if not run_script("src/train_model.py", "STEP 3: MODEL TRAINING"):
        print("[ERROR] Model training failed!")
        return
    
    print_header("PROJECT COMPLETED SUCCESSFULLY!")
    print("\nGenerated Files:")
    print("   |-- data/churn_frame.csv - Raw data")
    print("   |-- models/churn_model.joblib - Trained model")
    print("   |-- models/preprocessor.joblib - Data preprocessor")
    print("   |-- images/ - All visualization charts")
    print("   |-- outputs/ - Model comparison and predictions")
    
    print("\nKey Results:")
    print("   * Explore visualizations in the 'images' folder")
    print("   * Check 'outputs/model_comparison.csv' for model performance")
    
    # Optional: Start API server
    print("\n" + "="*70)
    start_api = input("Do you want to start the FastAPI server? (y/n): ")
    if start_api.lower() == 'y':
        print("\nStarting FastAPI server on http://localhost:8000")
        print("API Documentation: http://localhost:8000/docs")
        print("Press Ctrl+C to stop the server")
        subprocess.run([sys.executable, "-m", "uvicorn", "serving.app:app", "--host", "127.0.0.1", "--port", "8000", "--reload"])

if __name__ == "__main__":
    main()
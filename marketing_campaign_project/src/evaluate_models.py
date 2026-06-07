import joblib
import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

def evaluate_and_report():
    """
    Summarizes model performance from saved artifacts.
    """
    logging.info("Starting Final Model Evaluation...")

    # 1. Regression Summary
    reg_path = 'models/revenue_model.pkl'
    if os.path.exists(reg_path):
        data = joblib.load(reg_path)
        metrics = data['metrics']
        best_name = data['best_name']
        
        reg_df = pd.DataFrame(metrics).T.sort_values('R2', ascending=False)
        logging.info("\n--- Regression Model Comparison ---\n" + reg_df.to_string())
        logging.info(f"Winner: {best_name}")
    else:
        logging.warning("Regression model not found.")

    # 2. Classification Summary
    clf_path = 'models/profit_classifier.pkl'
    if os.path.exists(clf_path):
        data = joblib.load(clf_path)
        metrics = data['metrics']
        best_name = data['best_name']
        
        clf_df = pd.DataFrame(metrics).T.sort_values('F1', ascending=False)
        logging.info("\n--- Classification Model Comparison ---\n" + clf_df.to_string())
        logging.info(f"Winner: {best_name}")
    else:
        logging.warning("Classification model not found.")

if __name__ == "__main__":
    evaluate_and_report()

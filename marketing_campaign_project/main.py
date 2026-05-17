import os
import logging
import sys

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_cleaning import clean_data
from feature_engineering import engineer_features
from eda import perform_eda
from train_regression import train_regression
from train_classification import train_classification
from evaluate_models import evaluate_and_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting Marketing Campaign Analytics Pipeline...")

    # Define paths
    raw_data_path = 'data/raw/marketing_campaign.csv'
    cleaned_data_path = 'data/processed/cleaned_data.csv'
    engineered_data_path = 'data/processed/engineered_data.csv'
    reports_dir = 'reports'
    revenue_model_path = 'models/revenue_model.pkl'
    profit_model_path = 'models/profit_classifier.pkl'

    # Step 1 & 2: Data Cleaning
    logging.info("--- Phase: Data Cleaning ---")
    clean_data(raw_data_path, cleaned_data_path)

    # Step 3: Feature Engineering
    logging.info("--- Phase: Feature Engineering ---")
    engineer_features(cleaned_data_path, engineered_data_path)

    # Step 4: Exploratory Data Analysis
    logging.info("--- Phase: Exploratory Data Analysis ---")
    perform_eda(engineered_data_path, reports_dir)

    # Step 5: Regression Model
    logging.info("--- Phase: Regression Modeling ---")
    train_regression(engineered_data_path, revenue_model_path)

    # Step 6: Classification Model
    logging.info("--- Phase: Classification Modeling ---")
    train_classification(engineered_data_path, profit_model_path)

    # Step 7: Model Evaluation
    logging.info("--- Phase: Model Evaluation & Reporting ---")
    evaluate_and_report()

    logging.info("Pipeline executed successfully! You can now run the Streamlit app: `streamlit run app/streamlit_app.py`")

if __name__ == "__main__":
    main()

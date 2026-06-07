import os
import logging
import sys
from pathlib import Path

# Load centralized config
from config import (
    RAW_DATA_FILE, CLEANED_DATA_FILE, ENGINEERED_DATA_FILE,
    REVENUE_MODEL_FILE, PROFIT_CLASSIFIER_FILE,
    REPORTS_DIR, LOG_FILE,
    ensure_directories, validate_project_structure
)

# Fix for potential typo in config or missing import
try:
    from config import SRC_DIR
except ImportError:
    SRC_DIR = Path(__file__).resolve().parent / "src"

# Ensure src is in path
sys.path.append(str(SRC_DIR))

from src.data_cleaning import clean_data
from src.feature_engineering import engineer_features
from src.eda import perform_eda
from src.train_regression import train_regression
from src.train_classification import train_classification
from src.evaluate_models import evaluate_and_report

# Configure master logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def run_pipeline():
    logging.info("==================================================")
    logging.info("STARTING MARKETING INTELLIGENCE PIPELINE")
    logging.info("==================================================")

    # 0. Startup Validation
    logging.info("Step 0: Validating Project Structure...")
    struct_status = validate_project_structure()
    for name, stat in struct_status.items():
        logging.info(f"- {name}: {stat}")
    
    if "FAIL" in struct_status.values():
        logging.error("CRITICAL: Project structure is invalid. Check missing directories.")
        return

    # Ensure output directories exist
    ensure_directories()

    # Checklist for final report
    checklist = {
        "Data Merging": "PENDING",
        "Data Cleaning": "PENDING",
        "Feature Engineering": "PENDING",
        "EDA Reports": "PENDING",
        "Regression Modeling": "PENDING",
        "Classification Modeling": "PENDING",
        "System Evaluation": "PENDING"
    }

    # 1. Check for Raw Data
    if not RAW_DATA_FILE.exists():
        logging.warning(f"MISSING: {RAW_DATA_FILE}")
        logging.info("Fix: Run 'python merge_data.py' to generate the merged dataset.")
        return
    checklist["Data Merging"] = "PASS"

    # 2. Data Cleaning
    logging.info("PHASE 1: Data Validation & Cleaning...")
    try:
        clean_data(str(RAW_DATA_FILE), str(CLEANED_DATA_FILE))
        if CLEANED_DATA_FILE.exists():
            checklist["Data Cleaning"] = "PASS"
        else:
            raise FileNotFoundError("Cleaned data file was not created.")
    except Exception as e:
        logging.error(f"FAIL: Data Cleaning failed: {e}")
        return

    # 3. Feature Engineering
    logging.info("PHASE 2: Business Feature Engineering...")
    try:
        engineer_features(str(CLEANED_DATA_FILE), str(ENGINEERED_DATA_FILE))
        if ENGINEERED_DATA_FILE.exists():
            checklist["Feature Engineering"] = "PASS"
        else:
            raise FileNotFoundError("Engineered data file was not created.")
    except Exception as e:
        logging.error(f"FAIL: Feature Engineering failed: {e}")
        return

    # 4. EDA
    logging.info("PHASE 3: Exploratory Data Analysis & Visualization...")
    try:
        perform_eda(str(ENGINEERED_DATA_FILE), str(REPORTS_DIR))
        checklist["EDA Reports"] = "PASS"
    except Exception as e:
        logging.error(f"FAIL: EDA failed: {e}")
        # Continue anyway as modeling might still work

    # 5. Modeling - Regression
    logging.info("PHASE 4: Training Revenue Prediction Models...")
    try:
        train_regression(str(ENGINEERED_DATA_FILE), str(REVENUE_MODEL_FILE))
        if REVENUE_MODEL_FILE.exists():
            checklist["Regression Modeling"] = "PASS"
    except Exception as e:
        logging.error(f"FAIL: Regression training failed: {e}")

    # 6. Modeling - Classification
    logging.info("PHASE 5: Training Profitability Classification Models...")
    try:
        train_classification(str(ENGINEERED_DATA_FILE), str(PROFIT_CLASSIFIER_FILE))
        if PROFIT_CLASSIFIER_FILE.exists():
            checklist["Classification Modeling"] = "PASS"
    except Exception as e:
        logging.error(f"FAIL: Classification training failed: {e}")

    # 7. Evaluation
    logging.info("PHASE 6: System Evaluation & Aggregation...")
    try:
        evaluate_and_report()
        checklist["System Evaluation"] = "PASS"
    except Exception as e:
        logging.error(f"FAIL: Evaluation failed: {e}")

    logging.info("\n" + "="*50)
    logging.info("FINAL PIPELINE VALIDATION CHECKLIST")
    logging.info("="*50)
    for task, status in checklist.items():
        logging.info(f"{task:<25}: {status}")
    logging.info("="*50)
    
    if all(s == "PASS" for s in checklist.values()):
        logging.info("SUCCESS: Full Pipeline Executed Successfully!")
    else:
        logging.warning("COMPLETED: Pipeline finished with some warnings/failures.")
    
    logging.info(f"Streamlit Dashboard: streamlit run app/streamlit_app.py")
    logging.info("="*50)

if __name__ == "__main__":
    run_pipeline()

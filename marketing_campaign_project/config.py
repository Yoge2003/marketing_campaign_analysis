import os
from pathlib import Path

# Project Root (marketing_campaign_project/)
PROJECT_ROOT = Path(__file__).resolve().parent

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_SOURCE = DATA_DIR / "source"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"

# Core Project Folders
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SQL_DIR = PROJECT_ROOT / "sql"
APP_DIR = PROJECT_ROOT / "app"
SRC_DIR = PROJECT_ROOT / "src"

# Specific File Paths
RAW_DATA_FILE = DATA_RAW / "marketing_campaign.csv"
CLEANED_DATA_FILE = DATA_PROCESSED / "cleaned_data.csv"
ENGINEERED_DATA_FILE = DATA_PROCESSED / "engineered_data.csv"

REVENUE_MODEL_FILE = MODELS_DIR / "revenue_model.pkl"
PROFIT_CLASSIFIER_FILE = MODELS_DIR / "profit_classifier.pkl"

LOG_FILE = PROJECT_ROOT / "pipeline.log"

def ensure_directories():
    """Ensure all required project directories exist."""
    dirs = [DATA_SOURCE, DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, SQL_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def validate_project_structure():
    """Checks if critical directories exist and returns status dictionary."""
    status = {}
    critical_dirs = {
        "Data Source": DATA_SOURCE,
        "Data Raw": DATA_RAW,
        "Data Processed": DATA_PROCESSED,
        "Models": MODELS_DIR,
        "Reports": REPORTS_DIR,
        "App": APP_DIR,
        "Source Code": SRC_DIR
    }
    
    for name, path in critical_dirs.items():
        status[name] = "PASS" if path.exists() else "FAIL"
    
    return status

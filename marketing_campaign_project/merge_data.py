import pandas as pd
import os
import logging
from config import DATA_SOURCE, RAW_DATA_FILE, LOG_FILE, ensure_directories

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def merge_datasets():
    """
    Locates source datasets in data/source, merges them, and saves to data/raw.
    """
    ensure_directories()
    
    datasets = {
        'Nykaa': DATA_SOURCE / 'nykaa_campaign_data_with_nulls.csv',
        'Purplle': DATA_SOURCE / 'purplle_campaign_data_with_nulls.csv',
        'Tira': DATA_SOURCE / 'tira_campaign_data_with_nulls.csv'
    }

    dataframes = []
    expected_columns = None

    logging.info("==================================================")
    logging.info("STARTING DATA MERGING PHASE")
    logging.info(f"Looking for source files in: {DATA_SOURCE}")

    source_missing = False
    for brand, path in datasets.items():
        if not path.exists():
            logging.error(f"CRITICAL: Dataset not found for {brand}: {path}")
            source_missing = True
            continue
        
        try:
            df = pd.read_csv(path)
            logging.info(f"Loaded {brand} dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Column consistency check
            current_columns = set(df.columns)
            if expected_columns is None:
                expected_columns = current_columns
            else:
                if current_columns != expected_columns:
                    logging.warning(f"Column mismatch in {brand} dataset!")
            
            df['Brand'] = brand
            dataframes.append(df)
            
        except Exception as e:
            logging.error(f"Error loading {brand} dataset: {str(e)}")

    if source_missing:
        logging.critical("Some source datasets are missing. Please ensure all files are in data/source/")
        return False

    if not dataframes:
        logging.critical("No datasets were loaded. Merging aborted.")
        return False

    # Merge all datasets
    try:
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        logging.info(f"Merged Dataset Statistics:")
        logging.info(f"Total Rows: {combined_df.shape[0]}")
        logging.info(f"Total Columns: {combined_df.shape[1]}")
        
        combined_df.to_csv(RAW_DATA_FILE, index=False)
        logging.info(f"SUCCESS: Saved merged dataset to {RAW_DATA_FILE}")
        logging.info("==================================================")
        return True
        
    except Exception as e:
        logging.error(f"Error during merging or saving: {str(e)}")
        return False

if __name__ == "__main__":
    merge_datasets()

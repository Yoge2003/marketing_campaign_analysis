import pandas as pd
import numpy as np
import logging
import os

# Configure logging to inherit from main or pipeline log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

def clean_data(input_path, output_path):
    """
    Advanced data cleaning and validation for marketing campaign data.
    """
    logging.info(f"Loading raw data for cleaning from {input_path}")
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return None

    df = pd.read_csv(input_path)
    initial_rows = len(df)

    # 1. Basic Sanitization
    df.columns = df.columns.str.strip()
    df.drop_duplicates(inplace=True)
    logging.info(f"Removed {initial_rows - len(df)} duplicates.")

    # 2. Date Validation
    if 'Date' in df.columns:
        logging.info("Validating dates...")
        # Handle mixed formats if any, parse intelligently
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        invalid_dates = df['Date'].isna().sum()
        if invalid_dates > 0:
            logging.warning(f"Found {invalid_dates} invalid dates. Filling with most frequent date.")
            df['Date'] = df['Date'].fillna(df['Date'].mode()[0])

    # 3. Numeric Validation (Negative values)
    numeric_cols = ['Duration', 'Impressions', 'Clicks', 'Leads', 'Conversions', 'Revenue', 'Acquisition_Cost']
    for col in numeric_cols:
        if col in df.columns:
            # Detect negative values
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                logging.warning(f"Found {neg_count} negative values in {col}. Converting to absolute.")
                df[col] = df[col].abs()

    # 4. Logical Consistency (Clicks > Impressions, Conversions > Clicks etc.)
    # Note: In real data, sometimes this happens due to tracking issues, 
    # but for a project we should at least check.
    if 'Clicks' in df.columns and 'Impressions' in df.columns:
        invalid_clicks = (df['Clicks'] > df['Impressions']).sum()
        if invalid_clicks > 0:
            logging.warning(f"Found {invalid_clicks} rows where Clicks > Impressions. Capsizing Clicks to Impressions.")
            df['Clicks'] = np.where(df['Clicks'] > df['Impressions'], df['Impressions'], df['Clicks'])

    if 'Conversions' in df.columns and 'Clicks' in df.columns:
        invalid_conv = (df['Conversions'] > df['Clicks']).sum()
        if invalid_conv > 0:
            logging.warning(f"Found {invalid_conv} rows where Conversions > Clicks. Capsizing Conversions to Clicks.")
            df['Conversions'] = np.where(df['Conversions'] > df['Clicks'], df['Clicks'], df['Conversions'])

    # 5. Missing Value Handling
    logging.info("Analyzing and handling missing values...")
    # Calculate null percentages for logging
    null_pct = df.isnull().mean() * 100
    logging.info(f"Null Percentages:\n{null_pct[null_pct > 0].to_string()}")

    # Numeric: Use Median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    # Categorical: Use Mode or 'Unknown'
    for col in df.select_dtypes(include=['object']).columns:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna("Unknown")

    # 6. ROI Re-calculation & Validation
    # ROI = ((Revenue - Cost) / Cost) * 100
    logging.info("Recalculating ROI for consistency...")
    df['Acquisition_Cost'] = df['Acquisition_Cost'].replace(0, 0.01) # Avoid division by zero
    df['ROI'] = ((df['Revenue'] - df['Acquisition_Cost']) / df['Acquisition_Cost']) * 100

    # 7. Outlier Treatment (Capping at 1st and 99th percentile for stability)
    logging.info("Capping outliers at 1st and 99th percentiles...")
    for col in numeric_cols:
        if col in df.columns:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)

    # Final Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Cleaned data saved to {output_path} (Final Rows: {len(df)})")
    return df

if __name__ == "__main__":
    clean_data('data/raw/marketing_campaign.csv', 'data/processed/cleaned_data.csv')

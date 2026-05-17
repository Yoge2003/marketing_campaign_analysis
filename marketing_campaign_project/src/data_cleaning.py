import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_data(input_path, output_path):
    """
    Cleans the raw marketing campaign data.
    """
    logging.info(f"Loading raw data from {input_path}")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logging.error(f"File not found: {input_path}")
        return

    # 1. Column standardization (strip spaces)
    df.columns = df.columns.str.strip()

    # 2. Duplicate Removal
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    logging.info(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows.")

    # 3. Data type conversion
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # 4. Missing value treatment
    logging.info("Handling missing values...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Fill numeric missing values with median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values with mode
    for col in categorical_cols:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna("Unknown")

    # 5. Outlier Detection and Capping using IQR
    logging.info("Capping outliers using IQR method...")
    cols_to_cap = ['Duration', 'Impressions', 'Clicks', 'Leads', 'Conversions', 'Revenue', 'Acquisition_Cost']
    for col in cols_to_cap:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    # 6. ROI Validation
    # Formula: ROI = ((Revenue - Acquisition_Cost) / Acquisition_Cost) * 100
    logging.info("Validating and correcting ROI values...")
    if 'Revenue' in df.columns and 'Acquisition_Cost' in df.columns:
        # Avoid division by zero
        df['Acquisition_Cost'] = df['Acquisition_Cost'].replace(0, np.nan)
        calculated_roi = ((df['Revenue'] - df['Acquisition_Cost']) / df['Acquisition_Cost']) * 100
        # Fill zero acquisition cost with 0 ROI
        calculated_roi = calculated_roi.fillna(0)
        df['ROI'] = calculated_roi

    logging.info(f"Saving cleaned data to {output_path}")
    # Ensure processed directory exists
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info("Data cleaning completed successfully.")
    return df

if __name__ == "__main__":
    clean_data('data/raw/marketing_campaign.csv', 'data/processed/cleaned_data.csv')

import pandas as pd
import numpy as np
import logging
import os
import ast
from sklearn.preprocessing import MultiLabelBinarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

def engineer_features(input_path, output_path):
    """
    Advanced Feature Engineering for Marketing Intelligence.
    """
    logging.info(f"Loading cleaned data for feature engineering from {input_path}")
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return None

    df = pd.read_csv(input_path)

    # 1. Profit_Flag (Logic strictly ROI > 0)
    logging.info("Creating Profit_Flag (ROI > 0 threshold)...")
    if 'ROI' in df.columns:
        df['Profit_Flag'] = np.where(df['ROI'] > 0, 'Profit', 'Loss')
        logging.info(f"Profit_Flag distribution:\n{df['Profit_Flag'].value_counts().to_string()}")

    # 2. Multi-label Encoding for Channel_Used
    logging.info("Applying multi-label encoding for Channel_Used...")
    if 'Channel_Used' in df.columns:
        def parse_channels(val):
            if pd.isna(val) or val == 'Unknown':
                return []
            if isinstance(val, str):
                if val.startswith('['):
                    try: return ast.literal_eval(val)
                    except: pass
                # Clean and split common patterns
                cleaned = val.replace('&', ',').replace('+', ',')
                return [x.strip() for x in cleaned.split(',')]
            return val

        df['Channel_Used_List'] = df['Channel_Used'].apply(parse_channels)
        mlb = MultiLabelBinarizer()
        channel_encoded = mlb.fit_transform(df['Channel_Used_List'])
        channel_classes = [f"Channel_{c.replace(' ', '_')}" for c in mlb.classes_]
        channel_df = pd.DataFrame(channel_encoded, columns=channel_classes, index=df.index)
        df = pd.concat([df, channel_df], axis=1).drop(columns=['Channel_Used_List'])

    # 3. Core Marketing KPIs
    logging.info("Calculating advanced Marketing KPIs...")
    
    # Helper to avoid division by zero
    def safe_div(num, den):
        return np.where(den == 0, 0, num / den)

    # Requested Primary KPIs
    df['CTR'] = safe_div(df['Clicks'], df['Impressions'])
    df['Conversion_Rate'] = safe_div(df['Conversions'], df['Clicks'])
    df['Cost_Per_Lead'] = safe_div(df['Acquisition_Cost'], df['Leads'])
    df['Revenue_Per_Conversion'] = safe_div(df['Revenue'], df['Conversions'])

    # Requested Additional KPIs
    df['Cost_Per_Click'] = safe_div(df['Acquisition_Cost'], df['Clicks'])
    df['Lead_Conversion_Rate'] = safe_div(df['Conversions'], df['Leads'])
    df['Revenue_Per_Click'] = safe_div(df['Revenue'], df['Clicks'])
    df['Revenue_Per_Lead'] = safe_div(df['Revenue'], df['Leads'])
    
    # Profit Margin = (Revenue - Cost) / Revenue
    df['Profit_Margin'] = safe_div(df['Revenue'] - df['Acquisition_Cost'], df['Revenue'])
    
    # Marketing Efficiency Score (MES) - Harmonic balance of ROI and Conversion Rate
    # Normalize ROI first to a 0-1 scale for the score if possible, or use a weighted approach.
    # Simple MES: ROI * Conversion_Rate (Interaction feature)
    df['Marketing_Efficiency_Score'] = df['ROI'] * df['Conversion_Rate']

    # 4. Time-based Features
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

    # Clean up any potential INF or NAN from engineering
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Engineered data saved to {output_path}. Total features: {df.shape[1]}")
    return df

if __name__ == "__main__":
    engineer_features('data/processed/cleaned_data.csv', 'data/processed/engineered_data.csv')

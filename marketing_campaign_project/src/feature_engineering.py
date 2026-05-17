import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MultiLabelBinarizer
import ast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def engineer_features(input_path, output_path):
    """
    Applies feature engineering to the cleaned data.
    """
    logging.info(f"Loading cleaned data from {input_path}")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logging.error(f"File not found: {input_path}")
        return

    # 1. Profit_Flag
    logging.info("Creating Profit_Flag...")
    if 'ROI' in df.columns:
        # Check if there are any negative ROIs. If not, split by median to allow classification.
        if (df['ROI'] <= 0).sum() == 0:
            median_roi = df['ROI'].median()
            logging.info(f"No negative ROI found. Splitting Profit_Flag by median ROI ({median_roi:.2f}) to enable classification.")
            df['Profit_Flag'] = np.where(df['ROI'] > median_roi, 'Profit', 'Loss')
        else:
            df['Profit_Flag'] = np.where(df['ROI'] > 0, 'Profit', 'Loss')

    # 2. Multi-label Encoding for Channel_Used
    logging.info("Applying multi-label encoding for Channel_Used...")
    if 'Channel_Used' in df.columns:
        # Some fields might be strings like "Facebook, Instagram" or "['Facebook', 'Instagram']"
        # We need to ensure it's a list.
        def parse_channels(val):
            if pd.isna(val) or val == 'Unknown':
                return []
            if isinstance(val, str):
                # Try to parse string representation of list safely
                if val.startswith('['):
                    try:
                        return ast.literal_eval(val)
                    except:
                        pass
                # Otherwise assume comma-separated
                return [x.strip() for x in val.split(',')]
            return val

        df['Channel_Used_List'] = df['Channel_Used'].apply(parse_channels)
        
        mlb = MultiLabelBinarizer()
        channel_encoded = mlb.fit_transform(df['Channel_Used_List'])
        channel_classes = [f"Channel_{c}" for c in mlb.classes_]
        
        channel_df = pd.DataFrame(channel_encoded, columns=channel_classes, index=df.index)
        df = pd.concat([df, channel_df], axis=1)
        
        # We can drop the temporary list column
        df.drop(columns=['Channel_Used_List'], inplace=True)

    # 3. Additional Features
    logging.info("Creating additional performance features...")
    
    # Safely handle division by zero
    def safe_divide(numerator, denominator):
        return np.where(denominator == 0, 0, numerator / denominator)

    if 'Clicks' in df.columns and 'Impressions' in df.columns:
        df['CTR'] = safe_divide(df['Clicks'], df['Impressions'])

    if 'Conversions' in df.columns and 'Clicks' in df.columns:
        df['Conversion_Rate'] = safe_divide(df['Conversions'], df['Clicks'])

    if 'Acquisition_Cost' in df.columns and 'Leads' in df.columns:
        df['Cost_Per_Lead'] = safe_divide(df['Acquisition_Cost'], df['Leads'])

    if 'Revenue' in df.columns and 'Conversions' in df.columns:
        df['Revenue_Per_Conversion'] = safe_divide(df['Revenue'], df['Conversions'])

    logging.info(f"Saving engineered data to {output_path}")
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info("Feature engineering completed successfully.")
    return df

if __name__ == "__main__":
    engineer_features('data/processed/cleaned_data.csv', 'data/processed/engineered_data.csv')

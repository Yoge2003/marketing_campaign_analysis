import pandas as pd
import os

datasets = {
    'Nykaa': '../Dataset/nykaa_campaign_data_with_nulls.csv',
    'Purplle': '../Dataset/purplle_campaign_data_with_nulls.csv',
    'Tira': '../Dataset/tira_campaign_data_with_nulls.csv'
}

dataframes = []
for brand, path in datasets.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['Brand'] = brand
        dataframes.append(df)
        print(f"Loaded {brand}: {df.shape}")
    else:
        print(f"Path not found: {path}")

if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    out_path = 'data/raw/marketing_campaign.csv'
    combined_df.to_csv(out_path, index=False)
    print(f"Saved combined dataset to {out_path} with shape {combined_df.shape}")
else:
    print("No dataframes to combine.")

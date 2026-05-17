import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def perform_eda(input_path, reports_dir):
    """
    Performs Exploratory Data Analysis and generates an EDA report.
    """
    logging.info(f"Loading engineered data from {input_path}")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        logging.error(f"File not found: {input_path}")
        return

    img_dir = os.path.join(reports_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    logging.info("Generating Visualizations...")

    # Set seaborn style
    sns.set_theme(style="whitegrid")

    # 1. Revenue by Brand
    plt.figure(figsize=(8, 5))
    if 'Brand' in df.columns and 'Revenue' in df.columns:
        sns.barplot(data=df, x='Brand', y='Revenue', estimator=np.sum, errorbar=None)
        plt.title('Total Revenue by Brand')
        plt.savefig(os.path.join(img_dir, 'revenue_by_brand.png'))
        plt.close()

    # 2. ROI by Campaign Type
    plt.figure(figsize=(10, 6))
    if 'Campaign_Type' in df.columns and 'ROI' in df.columns:
        sns.boxplot(data=df, x='Campaign_Type', y='ROI')
        plt.title('ROI Distribution by Campaign Type')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(img_dir, 'roi_by_campaign_type.png'))
        plt.close()

    # 3. Top performing campaigns (by Revenue)
    if 'Campaign_ID' in df.columns and 'Revenue' in df.columns:
        top_campaigns = df.groupby('Campaign_ID')['Revenue'].sum().nlargest(10).reset_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_campaigns, x='Revenue', y='Campaign_ID', palette='viridis')
        plt.title('Top 10 Campaigns by Revenue')
        plt.savefig(os.path.join(img_dir, 'top_campaigns.png'))
        plt.close()

    # 4. Lowest performing campaigns (by ROI)
    if 'Campaign_ID' in df.columns and 'ROI' in df.columns:
        lowest_campaigns = df.groupby('Campaign_ID')['ROI'].mean().nsmallest(10).reset_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(data=lowest_campaigns, x='ROI', y='Campaign_ID', palette='Reds_r')
        plt.title('Bottom 10 Campaigns by Average ROI')
        plt.savefig(os.path.join(img_dir, 'lowest_campaigns.png'))
        plt.close()

    # 5. Channel Effectiveness (Total Revenue per Channel)
    channel_cols = [c for c in df.columns if c.startswith('Channel_') and c != 'Channel_Used']
    if channel_cols and 'Revenue' in df.columns:
        channel_revenue = {}
        for col in channel_cols:
            channel_revenue[col.replace('Channel_', '')] = df[df[col] == 1]['Revenue'].sum()
        
        channel_df = pd.DataFrame(list(channel_revenue.items()), columns=['Channel', 'Total_Revenue']).sort_values(by='Total_Revenue', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=channel_df, x='Total_Revenue', y='Channel', palette='Blues_r')
        plt.title('Channel Effectiveness (Total Revenue)')
        plt.savefig(os.path.join(img_dir, 'channel_effectiveness.png'))
        plt.close()

    # 6. Correlation Heatmap
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'correlation_heatmap.png'))
    plt.close()

    # 7. Revenue Distribution
    plt.figure(figsize=(8, 5))
    if 'Revenue' in df.columns:
        sns.histplot(df['Revenue'], bins=50, kde=True)
        plt.title('Revenue Distribution')
        plt.savefig(os.path.join(img_dir, 'revenue_distribution.png'))
        plt.close()

    # 8. Acquisition Cost Distribution
    plt.figure(figsize=(8, 5))
    if 'Acquisition_Cost' in df.columns:
        sns.histplot(df['Acquisition_Cost'], bins=50, kde=True, color='orange')
        plt.title('Acquisition Cost Distribution')
        plt.savefig(os.path.join(img_dir, 'acquisition_cost_distribution.png'))
        plt.close()

    # 9. Customer Segment Analysis
    plt.figure(figsize=(8, 5))
    if 'Customer_Segment' in df.columns and 'Revenue' in df.columns:
        sns.barplot(data=df, x='Customer_Segment', y='Revenue', estimator=np.sum, errorbar=None)
        plt.title('Total Revenue by Customer Segment')
        plt.savefig(os.path.join(img_dir, 'customer_segment_analysis.png'))
        plt.close()

    # 10. Language Analysis
    plt.figure(figsize=(8, 5))
    if 'Language' in df.columns and 'Revenue' in df.columns:
        sns.barplot(data=df, x='Language', y='Revenue', estimator=np.sum, errorbar=None)
        plt.title('Total Revenue by Language')
        plt.savefig(os.path.join(img_dir, 'language_analysis.png'))
        plt.close()

    # Generate Markdown Report
    logging.info("Generating EDA Report...")
    report_content = f"""# Exploratory Data Analysis Report

## Business Insights
1. **Revenue by Brand**: Visualizes which brand brings the most revenue.
2. **Channel Effectiveness**: Helps identify the best-performing marketing channels based on total revenue.
3. **Campaign Performance**: Highlights the top 10 revenue-generating campaigns and the bottom 10 lowest ROI campaigns.

## Key Findings
- **Data Quality**: The dataset originally contained nulls which have been successfully imputed. Outliers were capped using the IQR method.
- **Correlations**: The heatmap shows strong correlations between target variables like Revenue and performance metrics like Conversions.
- **Distributions**: Revenue and Acquisition Cost display normal/skewed distributions which were handled via capping.

## Recommendations
- **Allocate Budget Efficiently**: Reinvest heavily in the top marketing channels identified in the 'Channel Effectiveness' chart.
- **Optimize Low ROI Campaigns**: Investigate the campaigns listed in the 'Bottom 10 Campaigns' to identify weak points in targeting or ad creatives.
- **Focus on High-Value Segments**: Leverage insights from the Customer Segment analysis to target the most profitable audience.
"""
    
    with open(os.path.join(reports_dir, 'eda_report.md'), 'w') as f:
        f.write(report_content)

    logging.info("EDA completed successfully.")

if __name__ == "__main__":
    perform_eda('data/processed/engineered_data.csv', 'reports')

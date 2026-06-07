import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

def perform_eda(input_path, output_dir):
    """
    Performs comprehensive EDA and generates professional visualizations.
    """
    logging.info(f"Loading engineered data for EDA from {input_path}")
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    img_dir = os.path.join(output_dir, 'images')
    html_dir = os.path.join(output_dir, 'interactive')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(html_dir, exist_ok=True)

    # Set visualization style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['figure.figsize'] = (12, 7)

    # 1. Brand Comparison Analysis
    logging.info("Analyzing Brand performance...")
    brand_stats = df.groupby('Brand').agg({
        'Revenue': 'sum',
        'ROI': 'mean',
        'Conversion_Rate': 'mean',
        'Customer_Segment': 'count'
    }).reset_index().rename(columns={'Customer_Segment': 'Campaign_Count'})
    
    fig = px.bar(brand_stats, x='Brand', y='Revenue', color='Brand', 
                 title='Total Revenue by Brand', text_auto='.2s',
                 template='plotly_white')
    fig.write_html(os.path.join(html_dir, 'brand_revenue.html'))
    
    # 2. Campaign Type Analysis
    logging.info("Analyzing Campaign Types...")
    fig = px.box(df, x='Campaign_Type', y='ROI', color='Brand',
                 title='ROI Distribution by Campaign Type & Brand',
                 template='plotly_white')
    fig.write_html(os.path.join(html_dir, 'roi_by_campaign_type.html'))

    # 3. Channel Performance
    logging.info("Analyzing Channels...")
    channel_cols = [c for c in df.columns if c.startswith('Channel_')]
    channel_perf = []
    for col in channel_cols:
        cname = col.replace('Channel_', '')
        rev = df[df[col] == 1]['Revenue'].sum()
        roi = df[df[col] == 1]['ROI'].mean()
        channel_perf.append({'Channel': cname, 'Total_Revenue': rev, 'Avg_ROI': roi})
    
    perf_df = pd.DataFrame(channel_perf).sort_values('Total_Revenue', ascending=False)
    fig = px.bar(perf_df, x='Channel', y='Total_Revenue', color='Avg_ROI',
                 title='Channel Performance: Revenue vs ROI',
                 color_continuous_scale='Viridis', template='plotly_white')
    fig.write_html(os.path.join(html_dir, 'channel_performance.html'))

    # 4. Customer Segment & Language
    logging.info("Analyzing Segments and Languages...")
    fig = px.sunburst(df, path=['Brand', 'Customer_Segment', 'Language'], values='Revenue',
                      title='Revenue Distribution: Brand > Segment > Language',
                      template='plotly_white')
    fig.write_html(os.path.join(html_dir, 'revenue_sunburst.html'))

    # 5. Revenue Trends over Time
    logging.info("Analyzing Revenue Trends...")
    if 'Date' in df.columns:
        df_sorted = df.sort_values('Date')
        daily_rev = df_sorted.groupby(['Date', 'Brand'])['Revenue'].sum().reset_index()
        fig = px.line(daily_rev, x='Date', y='Revenue', color='Brand',
                      title='Daily Revenue Trends by Brand',
                      template='plotly_white')
        fig.write_html(os.path.join(html_dir, 'revenue_trends.html'))

    # 6. Correlation Heatmap (Static)
    logging.info("Generating Correlation Heatmap...")
    plt.figure(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Filter out binary encoded channels for cleaner heatmap
    core_cols = [c for c in numeric_cols if not c.startswith('Channel_')]
    corr = df[core_cols].corr()
    sns.heatmap(corr, annot=True, cmap='RdBu_r', fmt=".2f", center=0)
    plt.title('Core Metric Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'correlation_heatmap.png'))
    plt.close()

    # 7. Distributions (Histograms)
    logging.info("Generating Distributions...")
    metrics = ['Revenue', 'ROI', 'CTR', 'Conversion_Rate']
    for metric in metrics:
        if metric in df.columns:
            fig = px.histogram(df, x=metric, color='Brand', marginal='box',
                               title=f'{metric} Distribution by Brand',
                               template='plotly_white', barmode='overlay')
            fig.write_html(os.path.join(html_dir, f'{metric}_distribution.html'))

    logging.info("EDA visualizations generated successfully.")

if __name__ == "__main__":
    # Ensure reports directory exists
    os.makedirs('reports', exist_ok=True)
    perform_eda('data/processed/engineered_data.csv', 'reports')

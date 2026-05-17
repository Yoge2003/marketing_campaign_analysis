# Multi-Brand Marketing Campaign Performance Analysis and Prediction

## Project Overview
This project is an end-to-end Machine Learning pipeline that analyzes marketing campaign data from multiple brands (Nykaa, Purplle, Tira). It processes raw data, engineers new features (like Profit/Loss classification, multi-label channel encoding), performs Exploratory Data Analysis (EDA), and builds predictive models for Revenue and Profitability. Finally, it deploys these models and insights using a Streamlit application.

## Folder Structure
```
marketing_campaign_project/
│
├── data/                  # Contains raw and processed datasets
├── notebooks/             # Jupyter notebooks for EDA and experimentation
├── sql/                   # SQL Schema and Business Queries
├── src/                   # Source code (Cleaning, Engineering, Models)
├── models/                # Saved .pkl Machine Learning Models
├── app/                   # Streamlit Dashboard Application
├── reports/               # Markdown reports and generated EDA charts
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
└── main.py                # Main orchestration script
```

## Setup Instructions

1. **Clone the repository** or download the project files.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare Data**: Ensure `marketing_campaign.csv` is located in `data/raw/` (or run the merge script provided in the root to combine multiple brand datasets into one).

## How to Run

### Run the Full Pipeline
To execute data cleaning, feature engineering, EDA generation, model training, and evaluation sequentially, run:
```bash
python main.py
```
This will output cleaned data to `data/processed/`, visualizations and reports to `reports/`, and saved models to `models/`.

### Run the Streamlit Dashboard
To launch the interactive dashboard:
```bash
streamlit run app/streamlit_app.py
```

## SQL Queries
The `sql/` directory contains `schema.sql` for setting up the MySQL database and `business_queries.sql` to derive insights such as Top 10 Campaigns, Brand-wise Performance, and Monthly Revenue Trends.

import pandas as pd
import numpy as np
import logging
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_regression(input_path, model_output_path):
    """
    Trains regression models to predict Revenue.
    """
    logging.info(f"Loading data for regression from {input_path}")
    df = pd.read_csv(input_path)

    # Define features and target
    target = 'Revenue'
    # Drop features that cause data leakage (like ROI, Profit_Flag) or are irrelevant/strings
    drop_cols = ['Campaign_ID', 'Date', 'Revenue', 'ROI', 'Profit_Flag', 'Channel_Used']
    
    # Select numeric columns and encoded channel columns
    features = df.drop(columns=[c for c in drop_cols if c in df.columns])
    features = pd.get_dummies(features, drop_first=True) # Encode remaining categoricals like Brand, Campaign_Type, etc.
    
    # Handle missing values in features if any are left
    features = features.fillna(0)
    
    X = features
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42)
    }

    best_model = None
    best_r2 = -np.inf
    best_name = ""
    
    results = []

    for name, model in models.items():
        logging.info(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        
        results.append(f"**{name}**\n- RMSE: {rmse:.2f}\n- MAE: {mae:.2f}\n- R2: {r2:.2f}\n- MSE: {mse:.2f}\n")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name

    logging.info(f"Best model selected: {best_name} with R2: {best_r2:.2f}")

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    
    # Save the model and the columns used for training
    joblib.dump({'model': best_model, 'columns': X.columns.tolist()}, model_output_path)
    logging.info(f"Saved {best_name} model to {model_output_path}")

    # Save report
    os.makedirs('reports', exist_ok=True)
    with open('reports/regression_report.md', 'w') as f:
        f.write("# Regression Model Evaluation (Target: Revenue)\n\n")
        f.write("\n".join(results))
        f.write(f"\n## Selected Best Model\n**{best_name}** was selected because it achieved the highest R2 score of {best_r2:.2f}.\n")

if __name__ == "__main__":
    train_regression('data/processed/engineered_data.csv', 'models/revenue_model.pkl')

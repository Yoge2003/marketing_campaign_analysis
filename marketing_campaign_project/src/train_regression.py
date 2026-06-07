import pandas as pd
import numpy as np
import logging
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

def train_regression(input_path, model_output_path):
    """
    Trains and compares multiple regression models for Revenue prediction.
    """
    logging.info(f"Loading data for regression training from {input_path}")
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    # 1. Feature Selection & Leakage Prevention
    target = 'Revenue'
    # Core drop list
    drop_cols = ['Campaign_ID', 'Date', 'Revenue', 'ROI', 'Profit_Flag', 'Channel_Used']
    
    # Aggressive leakage metrics removal
    leakage_metrics = [
        'Revenue_Per_Conversion', 'Revenue_Per_Click', 'Revenue_Per_Lead', 
        'Profit_Margin', 'Marketing_Efficiency_Score', 'Profit_Flag_Numeric',
        'ROI_Capped', 'Revenue_Capped'
    ]
    
    current_drop = [c for c in drop_cols + leakage_metrics if c in df.columns]
    X = df.drop(columns=current_drop)
    
    # Ensure no name with 'Revenue' or 'ROI' remains in features
    illegal_keywords = ['revenue', 'roi', 'profit_flag', 'margin']
    for col in X.columns:
        if any(key in col.lower() for key in illegal_keywords):
            logging.warning(f"Removing suspicious feature for regression: {col}")
            X.drop(columns=[col], inplace=True)

    # Encode categorical variables
    X = pd.get_dummies(X, columns=['Brand', 'Campaign_Type', 'Target_Audience', 
                                   'Language', 'Customer_Segment'], drop_first=True)
    
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Training
    model_definitions = {
        'Linear Regression': {'model': LinearRegression(), 'params': {}},
        'Random Forest': {'model': RandomForestRegressor(random_state=42), 'params': {'n_estimators': [100], 'max_depth': [10, 20]}}
    }
    if XGB_AVAILABLE:
        model_definitions['XGBoost'] = {'model': XGBRegressor(random_state=42, n_jobs=-1), 'params': {'n_estimators': [100], 'learning_rate': [0.1]}}

    best_overall_model = None
    best_overall_score = -np.inf
    best_model_name = ""
    performance_metrics = {}

    for name, config in model_definitions.items():
        logging.info(f"Training {name}...")
        grid = GridSearchCV(config['model'], config['params'], cv=3, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        preds = best_model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        performance_metrics[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        
        if r2 > best_overall_score:
            best_overall_score = r2
            best_overall_model = best_model
            best_model_name = name

    # 3. Visualizations
    os.makedirs('reports/images', exist_ok=True)
    preds = best_overall_model.predict(X_test)
    
    # Feature Importance
    feat_importance_df = pd.DataFrame()
    if hasattr(best_overall_model, 'feature_importances_'):
        importances = best_overall_model.feature_importances_
        indices = np.argsort(importances)[-15:]
        
        # Save as DataFrame for ranked table
        feat_importance_df = pd.DataFrame({
            'Feature': [X.columns[i] for i in np.argsort(importances)[::-1]],
            'Importance': importances[np.argsort(importances)[::-1]]
        })

        plt.figure(figsize=(10, 8))
        plt.title(f'Top Revenue Drivers - {best_model_name}')
        plt.barh(range(len(indices)), importances[indices], align='center', color='#4e73df')
        plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig('reports/images/regression_feature_importance.png')
        plt.close()

    # Actual vs Predicted (Residuals)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, preds, alpha=0.5, color='#4e73df')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('Actual vs Predicted Revenue')
    plt.xlabel('Actual Revenue ($)')
    plt.ylabel('Predicted Revenue ($)')
    plt.grid(True, alpha=0.3)
    plt.savefig('reports/images/regression_residuals.png')
    plt.close()

    # Residual Distribution (Error Histogram)
    residuals = y_test - preds
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='#4e73df')
    plt.axvline(0, color='red', linestyle='--')
    plt.title('Residual Distribution (Error Analysis)')
    plt.xlabel('Prediction Error ($)')
    plt.savefig('reports/images/regression_error_dist.png')
    plt.close()

    # 4. Save
    joblib.dump({
        'model': best_overall_model,
        'columns': X.columns.tolist(),
        'metrics': performance_metrics,
        'best_name': best_model_name,
        'n_samples': len(df),
        'n_features': X.shape[1],
        'target': target,
        'feature_importance': feat_importance_df
    }, model_output_path)
    logging.info(f"Regression training complete. Winner: {best_model_name}")

if __name__ == "__main__":
    train_regression('data/processed/engineered_data.csv', 'models/revenue_model.pkl')

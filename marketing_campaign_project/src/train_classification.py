import pandas as pd
import numpy as np
import logging
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_classification(input_path, model_output_path):
    """
    Trains and compares multiple classification models for profitability prediction.
    """
    logging.info(f"Loading data for classification training from {input_path}")
    if not os.path.exists(input_path): return

    df = pd.read_csv(input_path)
    target = 'Profit_Flag'
    
    # 1. Leakage Protection
    # ABSOLUTE NO-GO: Any feature derived from ROI or Revenue
    drop_cols = ['Campaign_ID', 'Date', 'ROI', 'Revenue', 'Profit_Flag', 'Channel_Used']
    
    # These are derived from Revenue/Cost in a way that reveals Profitability directly
    leakage_metrics = [
        'Profit_Margin', 'Marketing_Efficiency_Score', 'Revenue_Per_Conversion', 
        'Revenue_Per_Click', 'Revenue_Per_Lead', 'ROI_Capped', 'Revenue_Capped'
    ]
    
    current_drop = [c for c in drop_cols + leakage_metrics if c in df.columns]
    X = df.drop(columns=current_drop)
    
    # Automated leakage detector
    illegal_keywords = ['revenue', 'roi', 'profit', 'margin', 'gain', 'return']
    for col in X.columns:
        if any(key in col.lower() for key in illegal_keywords):
            logging.error(f"LEAKAGE DETECTED: Feature '{col}' contains illegal keyword. Removing...")
            X.drop(columns=[col], inplace=True)

    # Encode
    X = pd.get_dummies(X, columns=['Brand', 'Campaign_Type', 'Target_Audience', 
                                   'Language', 'Customer_Segment'], drop_first=True)
    y = df[target].map({'Profit': 1, 'Loss': 0})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. Training
    model_definitions = {
        'Logistic Regression': {'model': LogisticRegression(max_iter=2000, class_weight='balanced'), 'params': {'C': [1.0]}},
        'Random Forest': {'model': RandomForestClassifier(class_weight='balanced', random_state=42), 'params': {'n_estimators': [100], 'max_depth': [10]}}
    }
    
    performance_metrics = {}
    best_overall_model = None
    best_overall_score = -1
    best_model_name = ""

    for name, config in model_definitions.items():
        logging.info(f"Training {name}...")
        grid = GridSearchCV(config['model'], config['params'], cv=3, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        preds = best_model.predict(X_test)
        
        performance_metrics[name] = {
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds, zero_division=0),
            'Recall': recall_score(y_test, preds, zero_division=0),
            'F1': f1_score(y_test, preds, zero_division=0),
            'ROC_AUC': roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
        }
        
        if performance_metrics[name]['F1'] > best_overall_score:
            best_overall_score = performance_metrics[name]['F1']
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
        plt.title(f'Top Profitability Drivers - {best_model_name}')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
        plt.tight_layout()
        plt.savefig('reports/images/classification_feature_importance.png')
        plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Profit'], yticklabels=['Loss', 'Profit'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('reports/images/classification_confusion_matrix.png')
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
        'class_dist': df[target].value_counts(normalize=True).to_dict(),
        'class_counts': df[target].value_counts().to_dict(),
        'feature_importance': feat_importance_df
    }, model_output_path)
    logging.info(f"Classification training complete. Winner: {best_model_name}")

if __name__ == "__main__":
    train_classification('data/processed/engineered_data.csv', 'models/profit_classifier.pkl')

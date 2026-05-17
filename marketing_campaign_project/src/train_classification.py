import pandas as pd
import numpy as np
import logging
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_classification(input_path, model_output_path):
    """
    Trains classification models to predict Profit_Flag.
    """
    logging.info(f"Loading data for classification from {input_path}")
    df = pd.read_csv(input_path)

    target = 'Profit_Flag'
    if target not in df.columns:
        logging.error("Profit_Flag column not found!")
        return

    # Drop data leakage features and non-predictive strings
    drop_cols = ['Campaign_ID', 'Date', 'ROI', 'Revenue', 'Profit_Flag', 'Channel_Used']
    
    features = df.drop(columns=[c for c in drop_cols if c in df.columns])
    features = pd.get_dummies(features, drop_first=True)
    features = features.fillna(0)
    
    X = features
    # Map target to binary
    y = df[target].apply(lambda x: 1 if x == 'Profit' else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42)
    }

    best_model = None
    best_f1 = -1
    best_name = ""
    
    results = []

    for name, model in models.items():
        logging.info(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds).tolist()
        
        results.append(f"**{name}**\n- Accuracy: {acc:.2f}\n- Precision: {prec:.2f}\n- Recall: {rec:.2f}\n- F1 Score: {f1:.2f}\n- Confusion Matrix: {cm}\n")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    logging.info(f"Best model selected: {best_name} with F1: {best_f1:.2f}")

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump({'model': best_model, 'columns': X.columns.tolist()}, model_output_path)
    logging.info(f"Saved {best_name} model to {model_output_path}")

    # Save report
    os.makedirs('reports', exist_ok=True)
    with open('reports/classification_report.md', 'w') as f:
        f.write("# Classification Model Evaluation (Target: Profit_Flag)\n\n")
        f.write("\n".join(results))
        f.write(f"\n## Selected Best Model\n**{best_name}** was selected because it achieved the highest F1 Score of {best_f1:.2f}.\n")

if __name__ == "__main__":
    train_classification('data/processed/engineered_data.csv', 'models/profit_classifier.pkl')

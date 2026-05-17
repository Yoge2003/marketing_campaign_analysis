# Final Model Evaluation Report

This report summarizes the performance of both Regression (Revenue Prediction) and Classification (Profit/Loss Prediction) models.

# Regression Model Evaluation (Target: Revenue)

**Linear Regression**
- RMSE: 151532.91
- MAE: 92356.00
- R2: 0.84
- MSE: 22962222602.81

**Random Forest**
- RMSE: 1395.87
- MAE: 800.40
- R2: 1.00
- MSE: 1948458.06

## Selected Best Model
**Random Forest** was selected because it achieved the highest R2 score of 1.00.


---

# Classification Model Evaluation (Target: Profit_Flag)

**Logistic Regression**
- Accuracy: 0.97
- Precision: 0.97
- Recall: 0.97
- F1 Score: 0.97
- Confusion Matrix: [[16296, 535], [540, 15962]]

**Decision Tree**
- Accuracy: 0.99
- Precision: 0.99
- Recall: 0.99
- F1 Score: 0.99
- Confusion Matrix: [[16597, 234], [220, 16282]]

**Random Forest**
- Accuracy: 0.98
- Precision: 0.98
- Recall: 0.98
- F1 Score: 0.98
- Confusion Matrix: [[16566, 265], [289, 16213]]

## Selected Best Model
**Decision Tree** was selected because it achieved the highest F1 Score of 0.99.


## Conclusion
The best performing models have been selected and saved in the `models/` directory for deployment in the Streamlit application.

import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_and_report():
    """
    Combines individual model reports into a final model_report.md
    """
    logging.info("Generating Final Model Report...")
    
    regression_content = ""
    if os.path.exists('reports/regression_report.md'):
        with open('reports/regression_report.md', 'r') as f:
            regression_content = f.read()

    classification_content = ""
    if os.path.exists('reports/classification_report.md'):
        with open('reports/classification_report.md', 'r') as f:
            classification_content = f.read()

    final_report = f"""# Final Model Evaluation Report

This report summarizes the performance of both Regression (Revenue Prediction) and Classification (Profit/Loss Prediction) models.

{regression_content}

---

{classification_content}

## Conclusion
The best performing models have been selected and saved in the `models/` directory for deployment in the Streamlit application.
"""
    
    with open('reports/model_report.md', 'w') as f:
        f.write(final_report)
        
    logging.info("Final model_report.md generated successfully.")

if __name__ == "__main__":
    evaluate_and_report()

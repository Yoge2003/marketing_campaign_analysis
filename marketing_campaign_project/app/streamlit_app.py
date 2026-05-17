import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# Set Streamlit page config
st.set_page_config(page_title="Marketing Campaign Predictor", layout="wide", page_icon="📈")

# Load models safely
@st.cache_resource
def load_models():
    rev_model, rev_cols, cls_model, cls_cols = None, None, None, None
    try:
        if os.path.exists('models/revenue_model.pkl'):
            rev_data = joblib.load('models/revenue_model.pkl')
            rev_model = rev_data['model']
            rev_cols = rev_data['columns']
            
        if os.path.exists('models/profit_classifier.pkl'):
            cls_data = joblib.load('models/profit_classifier.pkl')
            cls_model = cls_data['model']
            cls_cols = cls_data['columns']
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return rev_model, rev_cols, cls_model, cls_cols

rev_model, rev_cols, cls_model, cls_cols = load_models()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard & Insights", "Predict Performance", "Upload New Data"])

if page == "Dashboard & Insights":
    st.title("📈 Marketing Campaign Performance Dashboard")
    st.markdown("Explore key insights and visualizations from the historical campaign data.")
    
    img_dir = "reports/images"
    if os.path.exists(img_dir):
        images = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        
        if images:
            cols = st.columns(2)
            for i, img in enumerate(images):
                with cols[i % 2]:
                    image = Image.open(os.path.join(img_dir, img))
                    st.image(image, caption=img.replace('.png', '').replace('_', ' ').title(), use_container_width=True)
        else:
            st.info("No visualizations found. Please run the EDA script first.")
    else:
        st.info("Reports directory not found. Please run the EDA pipeline.")

elif page == "Predict Performance":
    st.title("🔮 Predict Campaign Performance")
    st.markdown("Enter campaign details to predict Revenue and Profit/Loss.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            duration = st.number_input("Duration (days)", min_value=1, value=30)
            impressions = st.number_input("Impressions", min_value=1, value=10000)
            clicks = st.number_input("Clicks", min_value=0, value=500)
            leads = st.number_input("Leads", min_value=0, value=50)
            
        with col2:
            conversions = st.number_input("Conversions", min_value=0, value=10)
            acq_cost = st.number_input("Acquisition Cost ($)", min_value=1.0, value=1000.0)
            engagement = st.number_input("Engagement Score", min_value=0.0, max_value=10.0, value=5.0)
            
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            if not rev_model or not cls_model:
                st.error("Models not loaded properly.")
            else:
                try:
                    # Create input dataframe
                    input_data = pd.DataFrame([{
                        'Duration': duration,
                        'Impressions': impressions,
                        'Clicks': clicks,
                        'Leads': leads,
                        'Conversions': conversions,
                        'Acquisition_Cost': acq_cost,
                        'Engagement_Score': engagement,
                        'CTR': clicks / impressions if impressions else 0,
                        'Conversion_Rate': conversions / clicks if clicks else 0,
                        'Cost_Per_Lead': acq_cost / leads if leads else 0
                    }])
                    
                    # Align with expected columns
                    rev_input = pd.DataFrame(columns=rev_cols)
                    rev_input = pd.concat([rev_input, input_data], ignore_index=True)
                    rev_input = rev_input.fillna(0)
                    
                    predicted_revenue = rev_model.predict(rev_input)[0]
                    
                    # For classification, include revenue per conversion if needed, but we keep it simple
                    cls_input = pd.DataFrame(columns=cls_cols)
                    cls_input = pd.concat([cls_input, input_data], ignore_index=True)
                    cls_input = cls_input.fillna(0)
                    
                    predicted_class = cls_model.predict(cls_input)[0]
                    predicted_label = "Profit" if predicted_class == 1 else "Loss"
                    
                    st.success("Prediction Successful!")
                    st.metric("Predicted Revenue", f"${predicted_revenue:,.2f}")
                    st.metric("Predicted Outcome", predicted_label)
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

elif page == "Upload New Data":
    st.title("📁 Upload New Campaign Data")
    st.markdown("Upload a CSV file with new campaign data to get batch predictions.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:", df.head())
            
            if st.button("Process & Predict"):
                st.info("Batch prediction is not fully implemented in this demo, but the file was read successfully!")
        except Exception as e:
            st.error(f"Error reading file: {e}")

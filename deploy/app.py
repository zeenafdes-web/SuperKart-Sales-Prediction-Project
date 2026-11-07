import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="SuperKart Sales Predictor", layout="centered")
st.title("🛒 SuperKart Sales Prediction App")
st.write("Predict total sales revenue for a specific product and store location.")

# Cache model downloading
@st.cache_resource
def load_model():
    repo_id = 'Zenorg/SuperKart-Sales-Prediction-Project'
    model_filename = 'best_sales_model.joblib'
    # Download from Space files
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename, repo_type='space')
    return joblib.load(model_path)

try:
    model = load_model()
    st.success("Model pipeline loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Form for inputs
with st.form("prediction_form"):
    st.subheader("Enter Product & Store Characteristics")
    
    # Inputs aligned to categorical and numerical variables used in training
    col1, col2 = st.columns(2)
    with col1:
        product_weight = st.number_input("Product Weight", min_value=0.0, value=12.5, step=0.1)
        product_sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
        product_type = st.selectbox("Product Type", [
            "Fruits and Vegetables", "Snack Foods", "Household", "Frozen Foods", 
            "Dairy", "Canned", "Baking Goods", "Health and Hygiene", 
            "Soft Drinks", "Meat", "Breads", "Hard Drinks", 
            "Others", "Starchy Foods", "Breakfast", "Seafood"
        ])
        product_mrp = st.number_input("Product MRP ($)", min_value=0.0, value=140.0, step=1.0)
        
    with col2:
        store_est_year = st.number_input("Store Establishment Year", min_value=1950, max_value=2025, value=1999, step=1)
        store_size = st.selectbox("Store Size", ["Medium", "Small", "High"])
        store_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])

    submit = st.form_submit_button("Predict Sales")

if submit:
    # Map inputs into a pandas DataFrame matching exact column expectations
    input_data = pd.DataFrame({
        'Product_Weight': [product_weight],
        'Product_Sugar_Content': [product_sugar],
        'Product_Allocated_Area': [product_allocated_area],
        'Product_Type': [product_type],
        'Product_MRP': [product_mrp],
        'Store_Establishment_Year': [store_est_year],
        'Store_Size': [store_size],
        'Store_Location_City_Type': [store_city_type],
        'Store_Type': [store_type]
    })
    
    # Make prediction using serialized pipeline (preprocessor automatically integrated)
    prediction = model.predict(input_data)[0]
    st.metric("Predicted Sales Revenue:", f"${prediction:,.2f}")

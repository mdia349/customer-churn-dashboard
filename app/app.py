import streamlit as st
import pandas as pd
import joblib

model = joblib.load('../models/churn_model.pkl')

st.title("Customer Churn Prediction Dashboard")

gender = st.selectbox("Gender", ["Male", "Female"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
monthly_charges = st.slider("Monthly Charges", 0, 150, 70)
tenure = st.slider("Tenure (Months)", 0, 72, 12)

input_data = pd.DataFrame({
    'gender':[gender],
    'Contract':[contract],
    'MonthlyCharges':[monthly_charges],
    'tenure':[tenure]
})

input_data_encoded = pd.get_dummies(input_data)
input_data_encoded = input_data_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

if st.button("Predict Churn"):
    prediction = model.predict_proba(input_data_encoded)[0][1]
    st.write(f"Churn likelihood: **{prediction:.2%}**")
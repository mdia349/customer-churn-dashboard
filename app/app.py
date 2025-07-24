import streamlit as st
import pandas as pd
import joblib

#Load model and feature list
model = joblib.load('models/churn_model.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')

st.title("📉 Customer Churn Prediction Dashboard")
st.write("Predict whether a telecom customer is likely to churn.")

#Sidebar inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing" , ["Yes", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
total_charges = st.slider("Total Charges", 0.0, 10000.0, 2000.0)

#Construct user input
user_input = pd.dataframe({
    'gender':[gender],
    'SeniorCitizen':[senior],
    'Partner':[partner],
    'Dependents':[dependents],
    'tenure':[tenure],
    'PhoneService':[phone_service],
    'MultipleLines':[multiple_lines],
    'InternetService':[internet_service],
    'OnlineSecurity':[online_security],
    'Contract':[contract],
    'PaperlessBilling':[paperless_billing],
    'PaymentMethod':[payment_method],
    'MonthlyCharges':[monthly_charges],
    'TotalCharges':[total_charges]
})

#One-hot encode to match training data
input_encoded = pd.get_dummies(user_input)
input_encoded = input_encoded.reindex(columns = feature_columns, fill_value = 0)

if st.button("Predict Churn"):
    prob = model.predict_proba(input_encoded)[0][1]
    st.metric(label = "Churn Probability", value = f"{prob:.2%}")
    if prob > 0.5:
        st.warning("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")
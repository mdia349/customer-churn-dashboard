import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    f1_score
)
import matplotlib.pyplot as plt
import numpy as np

# Load model, features, and threshold
model = joblib.load('models/best_model.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')
try:
    best_threshold = joblib.load('models/best_threshold.pkl')
except:
    best_threshold = 0.5

st.title("\U0001F4C9 Customer Churn Prediction Dashboard")
st.write("Use this tool to predict telecom customer churn and explore historical churn trends.")

# User Input Form
st.sidebar.header("\U0001F4CB Customer Attributes")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_text = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
senior = 1 if senior_text == "Yes" else 0
partner_text = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
partner = "Yes" if partner_text == "Yes" else "No"
dependents_text = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
dependents = "Yes" if dependents_text == "Yes" else "No"
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone_service_text = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
phone_service = "Yes" if phone_service_text == "Yes" else "No"
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing_text = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
paperless_billing = "Yes" if paperless_billing_text == "Yes" else "No"
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.sidebar.slider("Monthly Charges", 0.0, 150.0, 70.0)
total_charges = st.sidebar.slider("Total Charges", 0.0, 10000.0, 2000.0)


#  User input data
user_input = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [senior],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [phone_service],
    'MultipleLines': [multiple_lines],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
})

# Engineered features for user input data

user_input['IsMonthly'] = (user_input['Contract'] == 'Month-to-month').astype(int)
user_input['IsNewCustomer'] = (user_input['tenure'] < 6).astype(int)
user_input['LogTenure'] = np.log1p(user_input['tenure'])

contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
user_input['ContractLength'] = user_input['Contract'].map(contract_map)
user_input['TenureToContractRatio'] = user_input['tenure'] / user_input['ContractLength'].replace(0, 1)

user_input['IsElectronicCheck'] = (user_input['PaymentMethod'] == 'Electronic check').astype(int)
user_input['HasFiber'] = (user_input['InternetService'] == 'Fiber optic').astype(int)
user_input['FiberHighCharges'] = ((user_input['HasFiber'] == 1) & (user_input['MonthlyCharges'] > 80)).astype(int)
user_input['FiberWithoutSupport'] = (
    (user_input['InternetService'] == 'Fiber optic') &
    (user_input['OnlineSecurity'] == 'No')
).astype(int)

input_encoded = pd.get_dummies(user_input)
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# Prediction
st.header("\U0001F52E Churn Prediction")
threshold = st.slider("Prediction Threshold", 0.0, 1.0, float(best_threshold), 0.01)

if st.button("Predict Churn"):
    prob = model.predict_proba(input_encoded)[0][1]
    st.metric(label="Churn Probability", value=f"{prob:.2%}")
    if prob > threshold:
        st.warning("\u26A0\ufe0f This customer is likely to churn.")
    else:
        st.success("\u2705 This customer is likely to stay.")

# Load data for EDA
@st.cache_data
def load_data():
    df = pd.read_csv("data/telco_churn.csv")
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df['ChurnLabel'] = df['Churn'].map({0: 'Stayed', 1: 'Churned'})
    df['Senior'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    return df

df = load_data()

# EDA Section
st.header("\U0001F4CA Churn Trends and Data Exploration")

# Pie Chart
st.subheader("Churn Distribution")
fig_pie = px.pie(
    df, names='ChurnLabel',
    color='ChurnLabel',
    color_discrete_map={'Stayed': 'green', 'Churned': 'red'},
    title='Percentage of Customers Who Churned',
    hole=0.4
)
fig_pie.update_traces(textinfo='percent+label')
st.plotly_chart(fig_pie, use_container_width=True)

# Dropdown to select category
st.subheader("Churn by Category")
categorical_cols = [
    "gender", "Senior", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "Contract",
    "PaperlessBilling", "PaymentMethod"
]
selected_col = st.selectbox("Select a category:", categorical_cols)

fig_bar = px.histogram(
    df,
    x=selected_col,
    color='ChurnLabel',
    barmode='group',
    color_discrete_map={'Stayed': 'green', 'Churned': 'red'},
    title=f"Churn by {selected_col}"
)
fig_bar.update_layout(xaxis_title=selected_col, yaxis_title="Count")
st.plotly_chart(fig_bar, use_container_width=True)

# Box plot
st.subheader("Monthly Charges by Churn")
fig_box = px.box(
    df,
    x="ChurnLabel",
    y="MonthlyCharges",
    color="ChurnLabel",
    color_discrete_map={'Stayed': 'green', 'Churned': 'red'},
    labels={"ChurnLabel": "Churn"},
    title="Monthly Charges Distribution"
)
st.plotly_chart(fig_box, use_container_width=True)

#  model Evaluation Section
st.header("\U0001F9EA Model Evaluation")

@st.cache_data
def load_model_test_data():
    df = pd.read_csv("data/telco_churn.csv")
    df.drop(['customerID'], axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Feature engineering (same as training)
    df['IsMonthly'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['IsNewCustomer'] = (df['tenure'] < 6).astype(int)
    df['LogTenure'] = np.log1p(df['tenure'])
    contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
    df['ContractLength'] = df['Contract'].map(contract_map)
    df['TenureToContractRatio'] = df['tenure'] / df['ContractLength'].replace(0, 1)
    df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"])
    df['IsElectronicCheck'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    df['PaymentMethodRisk'] = df['PaymentMethod'].map({
        'Electronic check': 2,
        'Mailed check': 1,
        'Bank transfer (automatic)': 0,
        'Credit card (automatic)': 0
    })
    df['HasFiber'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['FiberHighCharges'] = ((df['HasFiber'] == 1) & (df['MonthlyCharges'] > 80)).astype(int)
    df['FiberWithoutSupport'] = (
        (df['InternetService'] == 'Fiber optic') &
        (df['TechSupport'] == 'No')
    ).astype(int)

    # Encode features to match training
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    from sklearn.model_selection import train_test_split
    X = df_encoded
    y = df['Churn']
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_test, X_val, y_test, y_val = load_model_test_data()
y_probs = model.predict_proba(X_val)[:, 1]
y_pred = (y_probs > best_threshold).astype(int)

# Metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

st.subheader("Model Performance Metrics")
st.write(f"**Threshold Used:** {best_threshold:.2f}")
st.write(f"**Accuracy:** {accuracy:.2%}")
st.write(f"**Precision:** {precision:.2%}")
st.write(f"**Recall:** {recall:.2%}")
st.write(f"**F1 Score:** {f1:.2%}")

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_val, y_pred)
fig_cm, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stayed", "Churned"])
disp.plot(ax=ax, cmap='Blues', colorbar=False)
st.pyplot(fig_cm)

# ROC Curve
st.subheader("ROC Curve and AUC")
fpr, tpr, thresholds = roc_curve(y_val, y_probs)
roc_auc = auc(fpr, tpr)
fig_roc, ax_roc = plt.subplots()
ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax_roc.set_xlim([0.0, 1.0])
ax_roc.set_ylim([0.0, 1.05])
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('Receiver Operating Characteristic')
ax_roc.legend(loc="lower right")
st.pyplot(fig_roc)

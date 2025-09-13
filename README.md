# 🔮 Customer Churn Prediction System

This project implements a customer churn prediction system using **Python**, **scikit-learn**, and **XGBoost**. The goal is to identify telecom customers likely to churn based on service usage, contract details, and demographics.

### 📊 Key highlights:
- **Data processing pipeline** with cleaning, feature engineering, and feature selection.
- **Model training & evaluation** with Logistic Regression, Random Forest, and XGBoost.
- **Hyperparameter tuning & threshold optimization** to maximize the F1 score.
- **Model persistence** for deployment-ready assets.
- **Streamlit app** for real-time churn prediction and visualization.

## 🛠️ Tech Stack:
- **Python 3.9+**
- **pandas, numpy** (data processing)
- **scikit-learn** (modeling, feature selection)
- **XGBoost** (gradient boosting model)
- **joblib** (model persistence)
- **Streamlit** (web app for predictions)
- **Jupyter Notebook** (EDA and analysis)

## 📂 Project Structure:

```bash
project-root/
│
├── app/
│   └── app.py                # Streamlit web app for predictions
│
├── data/
│   └── telco_churn.csv       # Input dataset
│
├── models/
│   ├── best_model.pkl        # Saved best-performing model
│   ├── best_threshold.pkl    # Optimal decision threshold
│   └── feature_columns.pkl   # Selected feature set
│
├── notebooks/
│   └── eda_churn_analysis.ipynb  # Exploratory Data Analysis
│
├── model_comparison.py       # Model training, tuning, and evaluation   
│
├── requirements.txt          # Project dependencies
│
└── README.md                 # Project documentation
```

## ⚙️ Installation:
**1. Clone this repository:**
```bash
git clone https://github.com/mdia349/customer-churn-dashboard.git
cd customer-churn-dashboard
```

**2. Create and activate a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

**3. Install all required modules from requirements.txt:**
```bash
pip install -r requirements.txt
```

## 🚀 Usage:

**1. Train & Evaluate Models:**
Run the model training script to compare algorithms and save the best model:
```bash
python scripts/model_comparison.py
```

**2. Launch the Streamlit App**
Start the app for interactive churn prediction:
```bash
streamlit run app/app.py
```

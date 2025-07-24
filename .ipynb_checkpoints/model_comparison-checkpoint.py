import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier  # optional, requires `pip install xgboost`

# Load and clean the data
df = pd.read_csv("data/telco_churn.csv")
df.drop(['customerID'], axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Optional: Feature engineering
df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"])
df['TotalServices'] = (
    df[['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies']] == 'Yes'
).sum(axis=1)

# Encode features
df_encoded = pd.get_dummies(df)

# Split data
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=5000),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
}

# Compare models with progress bar
results = []

print("Training and evaluating models...\n")

for name, model in tqdm(models.items(), desc="Comparing Models"):
    model.fit(X_train, y_train)
    
    y_probs = model.predict_proba(X_test)[:, 1]       # Get churn probabilities
    threshold = 0.5
    y_pred = (y_probs > threshold).astype(int)        # Apply custom threshold

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    })

# Show results
results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
print(results_df)
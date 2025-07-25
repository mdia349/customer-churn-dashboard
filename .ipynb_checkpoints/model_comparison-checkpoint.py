import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from tqdm import tqdm
import warnings
import joblib
import numpy as np
warnings.filterwarnings("ignore")

# Load and clean the data
df = pd.read_csv("data/telco_churn.csv")
df.drop(['customerID'], axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Feature engineering

df['IsMonthly'] = (df['Contract'] == 'Month-to-month').astype(int)

df['IsNewCustomer'] = (df['tenure'] < 6).astype(int)

df['LogTenure'] = np.log1p(df['tenure']) 

contract_map = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
df['ContractLength'] = df['Contract'].map(contract_map)
df['TenureToContractRatio'] = df['tenure'] / df['ContractLength'].replace(0, 1)

df['TenureGroup'] = pd.cut(
    df['tenure'],
    bins=[0, 12, 24, 48, 72],
    labels=["0-12", "13-24", "25-48", "49-72"]
)


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

# Encode features
df_encoded = pd.get_dummies(df)

# Split data
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']


from sklearn.feature_selection import SelectKBest, mutual_info_classif
# Apply feature selection to reduce noise
selector = SelectKBest(score_func=mutual_info_classif, k=30)  # pick top 30 features
X_selected = selector.fit_transform(X, y)

# Save column names for reporting if needed
selected_features = selector.get_support(indices=True)
selected_columns = X.columns[selected_features]

# Update X
X = pd.DataFrame(X_selected, columns=selected_columns)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Define hyperparameter grid for Logistic Regression
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs'],
    'penalty': ['l2'],
    'class_weight': ['balanced']
}

# Define hyperparameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']
}

# Define hyperparameter grid for XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'scale_pos_weight': [1, 2, 3]
}

# Define models with tuning
models = {
    "Logistic Regression (Tuned)": GridSearchCV(
        LogisticRegression(max_iter=10000),
        param_grid=param_grid_lr,
        cv=5,
        scoring='f1',
        n_jobs=-1
    ),
    "Random Forest (Tuned)": GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid_rf,
        cv=5,
        scoring='f1',
        n_jobs=-1
    ),
    "XGBoost (Tuned - Precision)": GridSearchCV(
        XGBClassifier(eval_metric='logloss'),
        param_grid=param_grid_xgb,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
}

# Compare models with progress bar
results = []
best_f1_score = 0
best_model_name = None
best_model_instance = None
best_model_thresh = 0.5

print("Training and evaluating models with hyperparameter tuning...\n")

for name, model in tqdm(models.items(), desc="Comparing Models"):
    model.fit(X_train, y_train)
    best_model = model.best_estimator_ if isinstance(model, GridSearchCV) else model
    y_probs = best_model.predict_proba(X_test)[:, 1]

    # Threshold sweep for best F1
    best_threshold = 0.5
    best_f1 = 0
    for t in np.arange(0.3, 0.71, 0.01):
        y_pred_thresh = (y_probs > t).astype(int)
        f1 = f1_score(y_test, y_pred_thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    y_pred_final = (y_probs > best_threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred_final)
    precision = precision_score(y_test, y_pred_final)
    recall = recall_score(y_test, y_pred_final)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": best_f1,
        "Best Threshold": best_threshold
    })

    if best_f1 > best_f1_score:
        best_f1_score = best_f1
        best_model_name = name
        best_model_instance = best_model
        best_model_thresh = best_threshold

    if isinstance(model, GridSearchCV):
        print(f"\nBest params for {name}: {model.best_params_}")
        print(f"Best threshold: {best_threshold:.2f} | Best F1: {best_f1:.4f}\n")

# Show results
results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
print(results_df)

# Save best model and features
if best_model_instance:
    joblib.dump(best_model_instance, 'models/best_model.pkl')
    joblib.dump(X.columns.tolist(), 'models/feature_columns.pkl')
    joblib.dump(best_model_thresh, 'models/best_threshold.pkl')
    print(f"\nSaved best model ({best_model_name}) with F1 Score: {best_f1_score:.4f}\n")

# from sklearn.inspection import permutation_importance
# result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
# important = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)
# print(important.head(10))
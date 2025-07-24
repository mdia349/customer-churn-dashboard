import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

#Load data
df = pd.read_csv('data/telco_churn.csv')

#Drop Customer ID - not needed
df.drop('customerID', axis = 1, inplace = True)

#Convert Total Charges to numeric
#Converting invalid values to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')

#Handle missing values
df.dropna(inplace = True)

#Encode target
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

#Identify categorical features
categorical_cols = df.select_dtypes(include = ['object']).columns

#One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns = categorical_cols)

#Split into train/test sets
X = df_encoded.drop('Churn', axis = 1)
y = df_encoded['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Train model
model = RandomForestClassifier(n_estimators = 100, random_state = 42)
model.fit(X_train, y_train)

#Save model and feature list
os.makedirs("models", exist_ok = True)
joblib.dump(model, 'models/churn_model.pkl')
joblib.dump(X.columns.tolist(), 'models/feature_columns.pkl')

print("Model and feature list saved.")
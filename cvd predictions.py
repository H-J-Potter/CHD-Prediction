import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import cv2

# Load Dataset
dataset = pd.read_csv("Framingham.csv")

# Data Preprocessing
dataset = dataset.drop_duplicates()
dataset = dataset.dropna()  # Removing rows with missing values

# Define input features and target variable
input_features = [
    'male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',
    'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
    'diaBP', 'BMI', 'heartRate', 'glucose'
]
target = 'TenYearCHD'

# Split the dataset into training and testing sets
train, test = train_test_split(dataset, shuffle=True, test_size=0.1, random_state=42)
train_y = train[target]
train_x = train[input_features]
test_y = test[target]
test_x = test[input_features]

# Standardize Data
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# Train Models
lr_model = LogisticRegression().fit(train_x, train_y)
rf_model = RandomForestClassifier(max_depth=25, n_estimators=300).fit(train_x, train_y)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(train_x, train_y)
knn_model = KNeighborsClassifier(n_neighbors=20).fit(train_x, train_y)
dt_model = DecisionTreeClassifier(max_depth=20).fit(train_x, train_y)

# Evaluate Models
model_accuracies = {
    'Logistic Regression': accuracy_score(test_y, lr_model.predict(test_x)),
    'Random Forest': accuracy_score(test_y, rf_model.predict(test_x)),
    'XGBoost': accuracy_score(test_y, xgb_model.predict(test_x)),
    'KNN': accuracy_score(test_y, knn_model.predict(test_x)),
    'Decision Tree': accuracy_score(test_y, dt_model.predict(test_x))
}

# Identify the best-performing model
best_model_name = max(model_accuracies, key=model_accuracies.get)
print("\nModel Accuracies:")
for model, acc in model_accuracies.items():
    print(f"{model}: {acc:.2f}")

print(f"\nBest Performing Model: {best_model_name}")

# Map model names to trained model objects
best_model = {
    'Logistic Regression': lr_model,
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'KNN': knn_model,
    'Decision Tree': dt_model
}[best_model_name]

# User Input for Prediction
print("\nEnter Patient Details for CHD Prediction:")

user_data = {}
all_features = input_features  # Ensure all input features are included

for feature in all_features:
    while True:
        try:
            value = float(input(f"Enter value for {feature}: "))
            user_data[feature] = value
            break
        except ValueError:
            print("Invalid input. Please enter a numerical value.")

# Create a DataFrame with user input
user_df = pd.DataFrame([user_data])

# Ensure feature order matches training data
user_df = user_df[input_features]

# Standardize the input data using the trained scaler
user_df_scaled = scaler.transform(user_df)

# Make Prediction
prediction = best_model.predict(user_df_scaled)
prediction_proba = best_model.predict_proba(user_df_scaled) if hasattr(
    best_model, 'predict_proba') else None

# Display Prediction
if prediction[0] == 1:
    print("\n🔴 The model predicts a HIGH RISK of developing CHD in 10 years.")
else:
    print("\n🟢 The model predicts a LOW RISK of developing CHD in 10 years.")

# Display Prediction Probability if available
if prediction_proba is not None:
    print(
        f"Prediction Probability: {prediction_proba[0][1]:.2f} (High Risk) | {prediction_proba[0][0]:.2f} (Low Risk)"
    )


 #Import the functions from extracting.py
 #main.py
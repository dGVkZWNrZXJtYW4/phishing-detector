import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib

# Step 1: Construct the path to the dataset
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(base_dir, "data", "phishing_data.csv")

# Load the dataset
emails = pd.read_csv(dataset_path)

# Step 2: Preprocess the data
# Drop irrelevant columns
features = emails.drop(columns=['id', 'CLASS_LABEL'])
labels = emails['CLASS_LABEL']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for predictions
scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
joblib.dump(scaler, scaler_path)
print("\nScaler saved to models/scaler.pkl")

# Step 3: Train the XGBoost model
# Convert to DMatrix (XGBoost's optimized data structure)
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# Set XGBoost parameters
params = {
    "objective": "binary:logistic",  # For binary classification
    "eval_metric": "logloss",       # Log loss as evaluation metric
    "learning_rate": 0.1,           # Step size shrinkage
    "max_depth": 6,                 # Maximum depth of a tree
    "n_estimators": 100             # Number of boosting rounds
}

# Train the model
model = xgb.train(params, dtrain, num_boost_round=100)

# Step 4: Evaluate the model
# Make predictions on the test set
y_pred_probs = model.predict(dtest)
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_probs]

# Print evaluation metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# Save the model
model_path = os.path.join(base_dir, "models", "xgboost_model.json")
model.save_model(model_path)
print("\nModel saved to models/xgboost_model.json")

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Construct the path to the dataset
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(base_dir, "data", "phishing_data.csv")

# Step 2: Load the dataset
emails = pd.read_csv(dataset_path)

# Step 3: Display dataset info
print("Dataset Sample:")
print(emails.head())
print("\nDataset Info:")
print(emails.info())
print("\nClass Distribution:")
print(emails['CLASS_LABEL'].value_counts())

# Step 4: Preprocess the data
# Drop irrelevant columns (e.g., 'id')
features = emails.drop(columns=['id', 'CLASS_LABEL'])
labels = emails['CLASS_LABEL']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nData preprocessing complete!")
print(f"Training data shape: {X_train_scaled.shape}")
print(f"Test data shape: {X_test_scaled.shape}")

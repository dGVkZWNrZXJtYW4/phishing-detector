import os
import pandas as pd

# Construct the path to the dataset
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(base_dir, "data", "phishing_data.csv")

# Load the dataset
emails = pd.read_csv(dataset_path)

# Display the first few rows
print("Dataset Sample:")
print(emails.head())

# Check dataset structure and missing values
print("\nDataset Info:")
print(emails.info())

# Check the distribution of labels
label_column = 'CLASS_LABEL'
print("\nClass Distribution:")
print(emails[label_column].value_counts())

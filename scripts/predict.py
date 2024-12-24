import pandas as pd
from tabulate import tabulate  # Install using pip install tabulate
import os
import xgboost as xgb
import joblib
import numpy as np

# Load the model and scaler
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "models", "xgboost_model.json")
scaler_path = os.path.join(base_dir, "models", "scaler.pkl")

model = xgb.Booster()
model.load_model(model_path)
scaler = joblib.load(scaler_path)

# List of feature columns (48 features as in the training dataset)
feature_columns = [
    'NumDots', 'SubdomainLevel', 'PathLevel', 'UrlLength', 'NumDash',
    'NumDashInHostname', 'AtSymbol', 'TildeSymbol', 'NumUnderscore',
    'NumPercent', 'NumQueryComponents', 'NumAmpersand', 'NumHash',
    'NumNumericChars', 'NoHttps', 'RandomString', 'IpAddress',
    'DomainInSubdomains', 'DomainInPaths', 'HttpsInHostname',
    'HostnameLength', 'PathLength', 'QueryLength', 'DoubleSlashInPath',
    'NumSensitiveWords', 'EmbeddedBrandName', 'PctExtHyperlinks',
    'PctExtResourceUrls', 'ExtFavicon', 'InsecureForms',
    'RelativeFormAction', 'ExtFormAction', 'AbnormalFormAction',
    'PctNullSelfRedirectHyperlinks', 'FrequentDomainNameMismatch',
    'FakeLinkInStatusBar', 'RightClickDisabled', 'PopUpWindow',
    'SubmitInfoToEmail', 'IframeOrFrame', 'MissingTitle',
    'ImagesOnlyInForm', 'SubdomainLevelRT', 'UrlLengthRT',
    'PctExtResourceUrlsRT', 'AbnormalExtFormActionR',
    'ExtMetaScriptLinkRT', 'PctExtNullSelfRedirectHyperlinksRT'
]

# Define a batch of example URLs and their corresponding features
sample_data = [
    {"url": "https://example-safe-site.com", "features": [
        1.0, 0.0, 3.0, 45.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 30.0, 0.0, 0.0, 0.0, 0.0,
        0.2, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0
    ]},
    {"url": "https://phishing-example.com", "features": [
        3.0, 1.0, 5.0, 72.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 21.0, 44.0, 0.0, 0.0, 0.0, 0.0,
        0.25, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 1.0
    ]},
    {"url": "http://another-legitimate-site.org", "features": [
        1.0, 0.0, 2.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 25.0, 0.0, 0.0, 0.0, 0.0,
        0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0, 0.5, -1.0, 0.0, 0.0, 2.0
    ]}
]

# Predict function for batch
def predict_batch(data):
    results = []
    for row in data:
        # Debugging: Check feature length and print it
        #print(f"URL: {row['url']} - Feature count: {len(row['features'])}")
        
        # Check the feature length
        #if len(row["features"]) != len(feature_columns):
        #    raise ValueError(f"Feature length mismatch for URL: {row['url']}. "
        #                     f"Expected {len(feature_columns)}, got {len(row['features'])}")
        
        # Convert features to DataFrame with column names to suppress warnings
        features = pd.DataFrame([row["features"]], columns=feature_columns)
        scaled_features = scaler.transform(features)  # Scale the features
        dmatrix = xgb.DMatrix(scaled_features)
        prediction_prob = model.predict(dmatrix)[0]
        prediction = "Phishing" if prediction_prob > 0.5 else "Legitimate"
        results.append({"url": row["url"], "prediction": prediction})
    return results

# Test the batch prediction
if __name__ == "__main__":
    predictions = predict_batch(sample_data)
    
    # Format and display results in a table
    df = pd.DataFrame(predictions)
    print(tabulate(df, headers="keys", tablefmt="pretty"))

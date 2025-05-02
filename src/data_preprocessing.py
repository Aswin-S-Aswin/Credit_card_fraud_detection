import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'creditcard.csv')
PREPROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), 'preprocessed_data.csv')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please download it from Kaggle and place it here.")
    data = pd.read_csv(path)
    return data

def preprocess_data(df):
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Scale 'Time' and 'Amount', rest are PCA components already scaled
    scaler = StandardScaler()
    X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

    # Handle class imbalance with SMOTE (optional here, usually done in training)
    # smote = SMOTE(random_state=42)
    # X_res, y_res = smote.fit_resample(X, y)

    return X, y, scaler

def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Preprocessing data...")
    X, y, scaler = preprocess_data(df)

    print("Saving preprocessed data...")
    processed = X.copy()
    processed['Class'] = y
    processed.to_csv(PREPROCESSED_DATA_PATH, index=False)

    print("Saving scaler...")
    joblib.dump(scaler, SCALER_PATH)

    print("Preprocessing complete.")


def preprocess_input(data, scaler):
    # Data is expected as a dict with keys matching the 30 features including 'Time' and 'Amount'
    # Convert to DataFrame for compatibility
    df = pd.DataFrame([data])
    
    # Ensure 'Time' and 'Amount' columns exist before scaling
    if 'Time' not in df.columns or 'Amount' not in df.columns:
        raise ValueError("Input data must include 'Time' and 'Amount' features.")

    # Scale 'Time' and 'Amount' features
    df[['Time', 'Amount']] = scaler.transform(df[['Time', 'Amount']])
    
    return df

if __name__ == "__main__":
    main()

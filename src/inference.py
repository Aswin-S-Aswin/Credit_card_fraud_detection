import os
import pandas as pd
import joblib
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model or scaler files not found. Please train the model first.")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def preprocess_input(data, scaler):
    # Data is expected as a dict with keys matching the 30 features including 'Time' and 'Amount'
    df = pd.DataFrame([data])
    
    # Ensure 'Time' and 'Amount' columns exist before scaling
    required_columns = ['Time', 'Amount']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Input data must include the '{col}' feature.")

    # Scale 'Time' and 'Amount' features
    df[required_columns] = scaler.transform(df[required_columns])
    return df

def main():
    print("Loading model and scaler...")
    model, scaler = load_model_and_scaler()

    # Example input transaction - user should replace with real transaction data
    example_transaction = {
        'Time': 86400,   # seconds from start (example)
        'V1': 0.1, 'V2': -1.2, 'V3': 0.3, 'V4': -0.4, 'V5': 0.5,
        'V6': -0.6, 'V7': 0.7, 'V8': -0.8, 'V9': 0.9, 'V10': -1.0,
        'V11': 1.1, 'V12': -1.2, 'V13': 1.3, 'V14': -1.4, 'V15': 1.5,
        'V16': -1.6, 'V17': 1.7, 'V18': -1.8, 'V19': 1.9, 'V20': -2.0,
        'V21': 2.1, 'V22': -2.2, 'V23': 2.3, 'V24': -2.4, 'V25': 2.5,
        'V26': -2.6, 'V27': 2.7, 'V28': -2.8, 'Amount': 100.0
    }

    print("Preprocessing input transaction...")
    processed_input = preprocess_input(example_transaction, scaler)

    print("Predicting fraud probability...")
    fraud_prob = model.predict_proba(processed_input)[:, 1][0]

    print(f"Fraud probability for the given transaction: {fraud_prob:.4f}")
    threshold = 0.5
    if fraud_prob >= threshold:
        print("WARNING: This transaction is likely fraudulent!")
        print("Result: Fraud Detected")
    else:
        print("This transaction appears legitimate.")
        print("Result: No Fraud Detected")

if __name__ == "__main__":
    main()
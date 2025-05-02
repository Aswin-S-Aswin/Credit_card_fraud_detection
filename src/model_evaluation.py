import os
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

PREPROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), 'preprocessed_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Please train the model first.")
    model = joblib.load(path)
    return model

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessed data not found at {path}. Please run data_preprocessing.py first.")
    df = pd.read_csv(path)
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y

def main():
    print("Loading model...")
    model = load_model(MODEL_PATH)

    print("Loading preprocessed data...")
    X, y = load_data(PREPROCESSED_DATA_PATH)

    # Evaluate on test set which is last 20% - here splitting similarly as training
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Predicting on test data...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

if __name__ == "__main__":
    main()

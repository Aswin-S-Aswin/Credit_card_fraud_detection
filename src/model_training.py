import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score
import joblib

PREPROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), 'preprocessed_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

def load_preprocessed_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessed data not found at {path}. Please run data_preprocessing.py first.")
    df = pd.read_csv(path)
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y

def train(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_res, y_train_res)
    return clf

def main():
    print("Loading preprocessed data...")
    X, y = load_preprocessed_data(PREPROCESSED_DATA_PATH)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training model...")
    model = train(X_train, y_train)

    print("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    print("Saving trained model...")
    joblib.dump(model, MODEL_PATH)

    print("Model training complete.")

if __name__ == "__main__":
    main()

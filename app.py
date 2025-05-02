# filepath: c:\Users\S. ASWIN\Documents\Naan Muthalvan\Credit_card_fraud_detection\app.py
from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
DATA_PREPROCESSING_SCRIPT = os.path.join(BASE_DIR, 'src', 'data_preprocessing.py')
MODEL_TRAINING_SCRIPT = os.path.join(BASE_DIR, 'src', 'model_training.py')
MODEL_EVALUATION_SCRIPT = os.path.join(BASE_DIR, 'src', 'model_evaluation.py')
INFERENCE_SCRIPT = os.path.join(BASE_DIR, 'src', 'inference.py')

def run_script(script_path):
    result = subprocess.run(['python', script_path], capture_output=True, text=True)
    if result.returncode != 0:
        return {"error": result.stderr.strip()}
    return {"output": result.stdout.strip()}

@app.route('/')
def home():
    return "Credit Card Fraud Detection API is running!"

@app.route('/preprocess', methods=['POST'])
def preprocess():
    return jsonify(run_script(DATA_PREPROCESSING_SCRIPT))

@app.route('/train', methods=['POST'])
def train():
    return jsonify(run_script(MODEL_TRAINING_SCRIPT))

@app.route('/evaluate', methods=['POST'])
def evaluate():
    return jsonify(run_script(MODEL_EVALUATION_SCRIPT))

@app.route('/inference', methods=['POST'])
def inference():
    return jsonify(run_script(INFERENCE_SCRIPT))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
import os
import subprocess

# Define paths to the scripts
BASE_DIR = os.path.dirname(__file__)
DATA_PREPROCESSING_SCRIPT = os.path.join(BASE_DIR, 'src', 'data_preprocessing.py')
MODEL_TRAINING_SCRIPT = os.path.join(BASE_DIR, 'src', 'model_training.py')
MODEL_EVALUATION_SCRIPT = os.path.join(BASE_DIR, 'src', 'model_evaluation.py')
INFERENCE_SCRIPT = os.path.join(BASE_DIR, 'src', 'inference.py')
INFERENCE_RESULT_FILE = os.path.join(BASE_DIR, 'inference_result.txt')

def run_script(script_path):
    """Run a Python script and return its output."""
    print(f"Running {script_path}...")
    result = subprocess.run(['python', script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error while running {script_path}:\n{result.stderr}")
        return {"status": "error", "output": result.stderr.strip()}
    print(result.stdout)
    return {"status": "success", "output": result.stdout.strip()}

def main():
    print("Starting the pipeline...")

    # Step 1: Preprocess the data
    print("\nStep 1: Data Preprocessing")
    preprocess_result = run_script(DATA_PREPROCESSING_SCRIPT)
    if preprocess_result["status"] == "error":
        print("Data preprocessing failed. Exiting...")
        return

    # Step 2: Train the model
    print("\nStep 2: Model Training")
    train_result = run_script(MODEL_TRAINING_SCRIPT)
    if train_result["status"] == "error":
        print("Model training failed. Exiting...")
        return

    # Step 3: Evaluate the model
    print("\nStep 3: Model Evaluation")
    evaluate_result = run_script(MODEL_EVALUATION_SCRIPT)
    if evaluate_result["status"] == "error":
        print("Model evaluation failed. Exiting...")
        return

    # Step 4: Run inference
    print("\nStep 4: Inference")
    inference_result = run_script(INFERENCE_SCRIPT)
    if inference_result["status"] == "error":
        print("Inference failed. Exiting...")
        return

    # Extract fraud probability and determine result
    output = inference_result["output"]
    fraud_prob = 0.0
    for line in output.splitlines():
        if "Fraud probability" in line:
            fraud_prob = float(line.split(":")[-1].strip())
            break

    threshold = 0.5
    result_message = "Fraud Detected" if fraud_prob >= threshold else "No Fraud Detected"

    # Save the result to a file
    with open(INFERENCE_RESULT_FILE, 'w') as f:
        f.write(result_message)

    print("\nPipeline execution complete.")
    print("\nSummary:")
    print("Inference Result:", result_message)

if __name__ == "__main__":
    main()
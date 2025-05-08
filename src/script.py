import os
import subprocess


BASE_DIR = os.path.dirname(__file__)
DATA_PREPROCESSING_SCRIPT = os.path.join(BASE_DIR, 'data_preprocessing.py')
MODEL_TRAINING_SCRIPT = os.path.join(BASE_DIR, 'model_training.py')
MODEL_EVALUATION_SCRIPT = os.path.join(BASE_DIR, 'model_evaluation.py')
INFERENCE_SCRIPT = os.path.join(BASE_DIR, 'inference.py')

def run_script(script_path):
    print(f"Running {script_path}...")
    result = subprocess.run(['python', script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error while running {script_path}:\n{result.stderr}")
        exit(1)
    print(result.stdout)

def main():
    print("Starting the pipeline...")
    
    # Preprocess the data
    run_script(DATA_PREPROCESSING_SCRIPT)
    
    # Train the model
    run_script(MODEL_TRAINING_SCRIPT)
    
    # Evaluate the model
    run_script(MODEL_EVALUATION_SCRIPT)
    
    # Run inference (optional, for testing purposes)
    print("\nYou can now run the inference script manually if needed:")
    print(f"python {INFERENCE_SCRIPT}")

if __name__ == "__main__":
    main()
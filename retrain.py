import shutil
import os
from config import CHECKPOINT_DIR, LOG_DIR

def clear_training_data():
    # Clear checkpoints directory
    if os.path.exists(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)
        print(f"Deleted {CHECKPOINT_DIR}")

    # Clear logs directory
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
        print(f"Deleted {LOG_DIR}")

    # Recreate these directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    print("Directories have been recreated")

if __name__ == "__main__":
    clear_training_data()
    print("Ready for retraining. Please run train.py to start a new training session.")

import shutil
import os
from config import CHECKPOINT_DIR, LOG_DIR

def clear_training_data():
    # Clear previous training data and recreate directories
    if os.path.exists(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)
    os.makedirs(CHECKPOINT_DIR)

    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR)

    print("Previous training data cleared.")

if __name__ == "__main__":
    clear_training_data()
    print("Ready for retraining. Run train.py to start a new training session.")
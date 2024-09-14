import os
import tensorflow as tf

# Data paths
INPUT_DIR = r"/content/drive/MyDrive/GAN_nocross/train"
OUTPUT_DIR = r"/content/drive/MyDrive/GAN_nocross/results"
TEST_DIR = r"/content/drive/MyDrive/GAN_nocross/test"
CHECKPOINT_DIR = r"/content/drive/MyDrive/GAN_nocross/checkpoints"

# Image parameters
IMAGE_SIZE = (201, 320, 300)  # Original image size
PATCH_SIZE = (64, 128, 128)   # 3D patch size

# Data augmentation parameters
ROTATION_RANGE = 15
SCALE_RANGE = (0.9, 1.1)
BRIGHTNESS_RANGE = (0.8, 1.2)

# Training parameters
BATCH_SIZE = 2  
EPOCHS = 300
LEARNING_RATE = 0.0002
EARLY_STOPPING_PATIENCE = 30

# Model parameters
GENERATOR_FILTERS = 32  # Keep unchanged or adjust as needed

# Evaluation parameters
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "noise2noise_final_model")

# Logging parameters
LOG_DIR = "logs"

# Visualization parameters
VISUALIZATION_SLICE = 16  # Z-axis slice index for visualization
VISUALIZATION_INTERVAL = 1  # Perform FFT visualization every 5 epochs

# Other parameters
RANDOM_SEED = 92

# Ensure necessary directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Set random seed
tf.random.set_seed(RANDOM_SEED)

# GPU configuration
GPUS = tf.config.experimental.list_physical_devices('GPU')
if GPUS:
    try:
        for gpu in GPUS:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {GPUS}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU available, using CPU")

# Model save format
MODEL_SAVE_FORMAT = "keras"  # Can be "keras" or "h5"

# Loss function weights
SSIM_WEIGHT = 0.3
L1_WEIGHT = 0.2
GRAD_WEIGHT = 0.1
NUCLEUS_WEIGHT = 0.1999
FFT_WEIGHT = 0.0001 

# Image preprocessing parameters
APPLY_ADAPTIVE_HISTOGRAM_EQUALIZATION = True

import os
import tensorflow as tf

# 数据路径
INPUT_DIR = r"/content/drive/MyDrive/GAN_nocross/train"
OUTPUT_DIR = r"/content/drive/MyDrive/GAN_nocross/results"
TEST_DIR = r"/content/drive/MyDrive/GAN_nocross/test"
CHECKPOINT_DIR = r"/content/drive/MyDrive/GAN_nocross/checkpoints"


# 图像参数
IMAGE_SIZE = (201, 320, 300)  # 原始图像大小
PATCH_SIZE = (64, 128, 128)   # 改回3D patch size

# 数据增强参数
ROTATION_RANGE = 15
SCALE_RANGE = (0.9, 1.1)
BRIGHTNESS_RANGE = (0.8, 1.2)

# 训练参数
BATCH_SIZE = 2  
EPOCHS = 300
LEARNING_RATE = 0.0002
EARLY_STOPPING_PATIENCE = 30

# 模型参数
GENERATOR_FILTERS = 32  # 保持不变或根据需要调整

# 评估参数
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "noise2noise_final_model")

# 日志参数
LOG_DIR = "logs"

# 可视化参数
VISUALIZATION_SLICE = 16  # 用于可视化的z轴切片索引
VISUALIZATION_INTERVAL = 1  # 每5个epoch进行一次FFT可视化

# 其他参数
RANDOM_SEED = 92

# 确保必要的目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 设置随机种子
tf.random.set_seed(RANDOM_SEED)

# GPU 配置
GPUS = tf.config.experimental.list_physical_devices('GPU')
if GPUS:
    try:
        for gpu in GPUS:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"使用 GPU: {GPUS}")
    except RuntimeError as e:
        print(e)
else:
    print("没有可用的 GPU，使用 CPU")

# 模型保存格式
MODEL_SAVE_FORMAT = "keras"  # 可以是 "keras" 或 "h5"

# 损失函数权重
SSIM_WEIGHT = 0.3
L1_WEIGHT = 0.2
GRAD_WEIGHT = 0.1
NUCLEUS_WEIGHT = 0.1999
FFT_WEIGHT = 0.0001 


# 图像预处理参数
APPLY_ADAPTIVE_HISTOGRAM_EQUALIZATION = True
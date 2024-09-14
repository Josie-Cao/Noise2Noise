import os
import tensorflow as tf
import numpy as np
import tifffile
from utils.image_processing import normalize_image, advanced_augment_image, crop_image
from config import PATCH_SIZE, BATCH_SIZE, INPUT_DIR, IMAGE_SIZE, RANDOM_SEED

class ZebrafishDataset:
    def __init__(self, input_dir):
        # Initialize ZebrafishDataset
        self.input_dir = input_dir
        self.file_list = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
        self.num_images = len(self.file_list)

    def load_and_preprocess(self):
        # Load and preprocess image pairs
        for filename in self.file_list:
            image_path = os.path.join(self.input_dir, filename)
            image = tifffile.imread(image_path)
            image = normalize_image(image)
            yield image, image  # Yield the same image twice for Noise2Noise training

    @tf.function
    def _preprocess_image_pair(self, img1, img2):
        # Preprocess image pair with augmentation and cropping
        seed = tf.random.experimental.stateless_split(tf.random.experimental.get_global_generator().make_seeds(2)[0], num=2)[0]
        
        img1 = advanced_augment_image(img1, seed)
        img2 = advanced_augment_image(img2, seed)
        
        img1 = crop_image(img1, PATCH_SIZE, seed)
        img2 = crop_image(img2, PATCH_SIZE, seed)
        
        return img1, img2

    def create_dataset(self):
        # Create final dataset
        dataset = tf.data.Dataset.from_generator(
            self.load_and_preprocess,
            output_signature=(
                tf.TensorSpec(shape=IMAGE_SIZE, dtype=tf.float32),
                tf.TensorSpec(shape=IMAGE_SIZE, dtype=tf.float32)
            )
        )
        
        dataset = dataset.map(self._preprocess_image_pair, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

def get_dataset(is_training=True, validation_split=0.2):
    # Get training or validation dataset
    full_dataset = ZebrafishDataset(INPUT_DIR).create_dataset()
    
    dataset_size = sum(1 for _ in full_dataset)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    
    if is_training:
        return full_dataset.take(train_size)
    else:
        return full_dataset.skip(train_size)
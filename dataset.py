import os
import tensorflow as tf
import numpy as np
import tifffile
from utils.image_processing import normalize_image, advanced_augment_image, crop_image
from config import PATCH_SIZE, BATCH_SIZE, INPUT_DIR, IMAGE_SIZE, RANDOM_SEED

class ZebrafishDataset:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        
    def load_and_preprocess(self):
        
        image_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.tif')])
        print(f"Number of .tif files found: {len(image_files)}")
        
        if len(image_files) == 0:
            raise ValueError(f"No .tif files found in directory {self.input_dir}")
        
        def load_image_pair(index):
            img1_path = os.path.join(self.input_dir, image_files[index])
            img2_path = os.path.join(self.input_dir, image_files[(index + 30) % len(image_files)])
            
            img1 = tifffile.imread(img1_path)
            img2 = tifffile.imread(img2_path)
            
            # Ensure image shape matches IMAGE_SIZE
            if img1.shape != IMAGE_SIZE or img2.shape != IMAGE_SIZE:
                raise ValueError(f"Image shape {img1.shape} does not match IMAGE_SIZE {IMAGE_SIZE}")
            
            img1 = tf.convert_to_tensor(img1, dtype=tf.float32)
            img2 = tf.convert_to_tensor(img2, dtype=tf.float32)

            
            # Normalize images
            img1 = normalize_image(img1)
            img2 = normalize_image(img2)

            
            # Add channel dimension
            img1 = tf.expand_dims(img1, axis=-1)
            img2 = tf.expand_dims(img2, axis=-1)
            
            return img1, img2
        
        dataset = tf.data.Dataset.from_tensor_slices(range(len(image_files)))
        dataset = dataset.map(lambda x: tf.py_function(load_image_pair, [x], [tf.float32, tf.float32]),
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda x, y: (tf.ensure_shape(x, IMAGE_SIZE + (1,)), tf.ensure_shape(y, IMAGE_SIZE + (1,))),
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self._preprocess_image_pair, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset
        
    @tf.function
    def _preprocess_image_pair(self, img1, img2):
        # Set random seed
        tf.random.set_seed(RANDOM_SEED)
        
        # Apply data augmentation with the same random parameters
        random_seed = tf.random.uniform([], minval=0, maxval=1000000, dtype=tf.int32)
        img1 = advanced_augment_image(img1, random_seed)
        img2 = advanced_augment_image(img2, random_seed)
        
        # Random cropping
        crop_seed = tf.random.uniform([], minval=0, maxval=1000000, dtype=tf.int32)
        img1_patch = crop_image(img1, PATCH_SIZE, crop_seed)
        img2_patch = crop_image(img2, PATCH_SIZE, crop_seed)
        
        # Ensure consistent shape
        img1_patch = tf.ensure_shape(img1_patch, PATCH_SIZE + (1,))
        img2_patch = tf.ensure_shape(img2_patch, PATCH_SIZE + (1,))
        
        return img1_patch, img2_patch
    
    def create_dataset(self):
        dataset = self.load_and_preprocess()
        dataset = dataset.shuffle(buffer_size=100, seed=RANDOM_SEED)
        dataset = dataset.batch(BATCH_SIZE)
        return dataset.prefetch(tf.data.AUTOTUNE)

def get_dataset(is_training=True, validation_split=0.2):
    full_dataset = ZebrafishDataset(INPUT_DIR).create_dataset()
    dataset_size = tf.data.experimental.cardinality(full_dataset).numpy()
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size


    if is_training:
        train_dataset = full_dataset.take(train_size)
        return train_dataset
    else:
        val_dataset = full_dataset.skip(train_size)
        return val_dataset

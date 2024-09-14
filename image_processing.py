import tensorflow as tf
import numpy as np
from config import ROTATION_RANGE, SCALE_RANGE, BRIGHTNESS_RANGE, IMAGE_SIZE
from scipy.ndimage import map_coordinates, gaussian_filter

def normalize_image(image):
    # Normalize image to [0, 1] range
    return tf.cast(image, tf.float32) / 255.0

def denormalize_image(image):
    # Denormalize image from [0, 1] to [0, 255] range
    return tf.cast(tf.clip_by_value(image * 255.0, 0, 255), tf.uint8)

@tf.function
def advanced_augment_image(image, seed):
    # Apply advanced image augmentation
    image = tf.image.random_brightness(image, max_delta=0.2, seed=seed)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2, seed=seed)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    return image

@tf.function
def crop_image(image, patch_size, seed):
    # Randomly crop image to patch size
    return tf.image.random_crop(image, size=patch_size + (1,), seed=seed)

def pad_image(image, target_size):
    # Pad 3D single-channel image to target size
    current_shape = tf.shape(image)
    pad_size = target_size - current_shape
    paddings = tf.stack([
        [pad_size[0] // 2, pad_size[0] - pad_size[0] // 2],
        [pad_size[1] // 2, pad_size[1] - pad_size[1] // 2],
        [pad_size[2] // 2, pad_size[2] - pad_size[2] // 2],
        [0, 0]
    ])
    return tf.pad(image, paddings, mode='constant', constant_values=0)

def unpad_image(image, original_size):
    # Unpad 3D single-channel image to original size
    current_shape = tf.shape(image)
    start = (current_shape - original_size) // 2
    return tf.slice(image, start, original_size)

def apply_window(image, window_center, window_width):
    # Apply window level adjustment
    min_value = window_center - window_width // 2
    max_value = window_center + window_width // 2
    return tf.clip_by_value(image, min_value, max_value)

def adaptive_histogram_equalization(image, clip_limit=0.03):
    # Apply adaptive histogram equalization
    def clahe(img):
        img = tf.image.adjust_contrast(img, 2)
        img = tf.image.per_image_standardization(img)
        return tf.clip_by_value(img, 0, 1)
    
    image = tf.map_fn(clahe, image)
    return image

def add_gaussian_noise(image, mean=0, stddev=0.1):
    # Add Gaussian noise to the image
    noise = tf.random.normal(shape=tf.shape(image), mean=mean, stddev=stddev)
    noisy_image = image + noise
    return tf.clip_by_value(noisy_image, 0, 1)
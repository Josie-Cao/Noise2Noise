import tensorflow as tf
import numpy as np
import tifffile
import argparse
import os
from models.generator import build_unet_generator
from config import GENERATOR_FILTERS, PATCH_SIZE, IMAGE_SIZE, CHECKPOINT_PATH
from utils.image_processing import normalize_image, denormalize_image, pad_image, unpad_image

def load_model(checkpoint_path):
    # Load trained Noise2Noise model
    generator = build_unet_generator(PATCH_SIZE + (1,), GENERATOR_FILTERS)
    generator.load_weights(checkpoint_path)
    return generator

def process_volume(generator, volume):
    # Process a single 3D volume
    original_shape = volume.shape
    padded_volume = pad_image(volume, PATCH_SIZE + (1,))
    normalized_volume = normalize_image(padded_volume)
    
    # Add batch dimension
    input_volume = tf.expand_dims(normalized_volume, axis=0)
    
    # Apply generator
    denoised_volume = generator(input_volume, training=False)
    
    # Remove batch dimension
    denoised_volume = tf.squeeze(denoised_volume, axis=0)
    
    # Denormalize and remove padding
    denoised_volume = denormalize_image(denoised_volume)
    denoised_volume = unpad_image(denoised_volume, original_shape)
    
    return denoised_volume

def apply_noise2noise(input_path, output_path):
    # Apply Noise2Noise model to 3D+T image
    generator = load_model(CHECKPOINT_PATH)
    
    # Read input image
    input_image = tifffile.imread(input_path)
    
    # Process each time point
    denoised_volumes = []
    for t in range(input_image.shape[0]):
        volume = input_image[t]
        denoised_volume = process_volume(generator, volume)
        denoised_volumes.append(denoised_volume)
    
    # Stack denoised volumes into 4D image
    denoised_image = np.stack(denoised_volumes, axis=0)
    
    # Save result
    tifffile.imwrite(output_path, denoised_image)
    print(f"Denoised image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Noise2Noise model to denoise 3D+T image")
    parser.add_argument("input_path", type=str, help="Path to input image")
    parser.add_argument("output_path", type=str, help="Path to output image")
    args = parser.parse_args()
    
    apply_noise2noise(args.input_path, args.output_path)

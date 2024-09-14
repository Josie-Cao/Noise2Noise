import tensorflow as tf
import numpy as np
from models.generator import build_unet_generator
from data.dataset import get_dataset
from utils.image_processing import normalize_image, denormalize_image, add_gaussian_noise
from config import *
import matplotlib.pyplot as plt
import glob
import os

def load_model(checkpoint_path):
    # Load trained Noise2Noise model
    print(f"Attempting to load model from path: {checkpoint_path}")
    print(f"File exists: {os.path.exists(checkpoint_path)}")
    
    try:
        # Try to load the entire model
        model = tf.keras.models.load_model(checkpoint_path)
        print("Successfully loaded complete model")
        return model
    except:
        print("Unable to load complete model, attempting to load weights only")
        generator = build_unet_generator(PATCH_SIZE + (1,), GENERATOR_FILTERS)
        generator.compile(optimizer='adam', loss='mse')
        
        try:
            generator.load_weights(checkpoint_path)
            print("Successfully loaded model weights")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Will use untrained model for evaluation")
    
    return generator

def calculate_psnr(img1, img2):
    # Calculate PSNR
    return tf.image.psnr(img1, img2, max_val=1.0).numpy()

def calculate_ssim(img1, img2):
    # Calculate SSIM
    return tf.image.ssim(img1, img2, max_val=1.0).numpy()

def evaluate(test_dataset, generator):
    # Evaluate model performance
    psnr_scores = []
    ssim_scores = []

    for data in test_dataset:
        # Check if data is a tuple
        if isinstance(data, tuple):
            clean_images = data[0]  # Assume the first element is the input image
        else:
            clean_images = data

        # Ensure correct input shape
        if tf.rank(clean_images) == 4 and clean_images.shape[0] == 2:
            # If shape is [2,32,128,128], we only use the first image
            clean_images = clean_images[0]
        
        if tf.rank(clean_images) == 3:
            clean_images = tf.expand_dims(clean_images, axis=-1)  # Add channel dimension
        
        # Ensure shape is (32, 128, 128, 1)
        clean_images = tf.ensure_shape(clean_images, PATCH_SIZE + (1,))
        
        noisy_images = add_gaussian_noise(clean_images)
        
        # Add batch dimension
        noisy_images = tf.expand_dims(noisy_images, axis=0)
        
        denoised_images = generator(noisy_images, training=False)

        # Remove batch dimension and ensure consistent shape
        denoised_images = tf.squeeze(denoised_images)
        if tf.rank(denoised_images) == 3:
            denoised_images = tf.expand_dims(denoised_images, axis=-1)

        # Ensure both images have the same shape
        clean_images = tf.ensure_shape(clean_images, PATCH_SIZE + (1,))
        denoised_images = tf.ensure_shape(denoised_images, PATCH_SIZE + (1,))

        # Calculate metrics
        psnr = calculate_psnr(clean_images, denoised_images)
        ssim = calculate_ssim(clean_images, denoised_images)

        psnr_scores.append(psnr)
        ssim_scores.append(ssim)

    return np.mean(psnr_scores), np.mean(ssim_scores)

def visualize_results(clean_image, noisy_image, denoised_image, save_path):
    # Visualize denoising results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(clean_image[0, :, :, VISUALIZATION_SLICE, 0], cmap='gray')
    axes[0].set_title('Clean Image')
    axes[0].axis('off')
    
    axes[1].imshow(noisy_image[0, :, :, VISUALIZATION_SLICE, 0], cmap='gray')
    axes[1].set_title('Noisy Image')
    axes[1].axis('off')
    
    axes[2].imshow(denoised_image[0, :, :, VISUALIZATION_SLICE, 0], cmap='gray')
    axes[2].set_title('Denoised Image')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def find_best_weight_file(checkpoint_dir):
    # Find the best weight file in checkpoint directory
    best_weight_files = glob.glob(os.path.join(checkpoint_dir, "noise2noise_best_*.weights.h5"))
    if best_weight_files:
        # Sort by file modification time, choose the latest
        return max(best_weight_files, key=os.path.getmtime)
    
    # If no best weight file found, return the last weight file
    final_weight_file = os.path.join(checkpoint_dir, "noise2noise_final.weights.h5")
    if os.path.exists(final_weight_file):
        print("Warning: Best weight file not found, using weights from the last epoch.")
        return final_weight_file
    
    print(f"Error: No weight file found in directory {checkpoint_dir}.")
    return None

def main():
    # Main evaluation function
    # Check if weight file exists
    best_weight_file = find_best_weight_file(CHECKPOINT_DIR)
    if not best_weight_file:
        print(f"Error: No weight file found in directory: {CHECKPOINT_DIR}")
        return

    # Load test dataset
    test_dataset = get_dataset(is_training=False)

    # Load model (using best weight file)
    generator = load_model(best_weight_file)

    # Evaluate model
    psnr, ssim = evaluate(test_dataset, generator)
    print(f"Average PSNR: {psnr:.2f}")
    print(f"Average SSIM: {ssim:.4f}")

    # Visualize some results
    for clean_image in test_dataset.take(1):
        noisy_image = add_gaussian_noise(clean_image)
        denoised_image = generator(noisy_image, training=False)
        visualize_results(clean_image, noisy_image, denoised_image, "evaluation_results_final.png")

if __name__ == "__main__":
    main()
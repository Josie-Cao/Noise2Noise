import tensorflow as tf
import numpy as np
import os
import tifffile
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from models.generator import build_unet_generator
from config import GENERATOR_FILTERS, PATCH_SIZE, CHECKPOINT_DIR

def load_model_safely(model_path, weights_path=None):
    try:
        if not os.path.exists(model_path):
            print(f"model file doesn't exist : {model_path}")
            return None

        model = load_model(model_path, compile=False)
        print("load model successfully")

        if weights_path and os.path.exists(weights_path):
            model.load_weights(weights_path)
            print("Load weights successfully")
        
        return model
    except Exception as e:
        print(f"model load fail: {str(e)}")
        return None

def denoise_image(input_image, generator):
    if input_image.ndim == 3:
        input_image = np.expand_dims(input_image, axis=0)
    
    _, depth, height, width = input_image.shape
    expected_depth, expected_height, expected_width = 64, 128, 128
    
    depth_blocks = int(np.ceil(depth / expected_depth))
    height_blocks = int(np.ceil(height / expected_height))
    width_blocks = int(np.ceil(width / expected_width))
    
    output_image = np.zeros_like(input_image)
    
    for d in range(depth_blocks):
        for h in range(height_blocks):
            for w in range(width_blocks):
                d_start = d * expected_depth
                d_end = min((d + 1) * expected_depth, depth)
                h_start = h * expected_height
                h_end = min((h + 1) * expected_height, height)
                w_start = w * expected_width
                w_end = min((w + 1) * expected_width, width)
                
                block = input_image[:, d_start:d_end, h_start:h_end, w_start:w_end]
                
                if block.shape[1:] != (expected_depth, expected_height, expected_width):
                    pad_depth = expected_depth - block.shape[1]
                    pad_height = expected_height - block.shape[2]
                    pad_width = expected_width - block.shape[3]
                    block = np.pad(block, ((0, 0), (0, pad_depth), (0, pad_height), (0, pad_width)), mode='constant')
                
                block = np.expand_dims(block, axis=-1)
                block = block.astype(np.float32) / 255.0
                
                denoised_block = generator.predict(block)
                
                denoised_block = (denoised_block * 255.0).astype(input_image.dtype)
                denoised_block = np.squeeze(denoised_block)
                
                output_image[:, d_start:d_end, h_start:h_end, w_start:w_end] = denoised_block[:d_end-d_start, :h_end-h_start, :w_end-w_start]
    
    output_image = np.squeeze(output_image, axis=0)
    
    return output_image

def plot_comparison(original, denoised):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    mid_slice = original.shape[0] // 2
    
    ax1.imshow(original[mid_slice], cmap='gray')
    ax1.set_title('origin image')
    ax1.axis('off')
    
    ax2.imshow(denoised[mid_slice], cmap='gray')
    ax2.set_title('denoised image')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model_path = os.path.join(CHECKPOINT_DIR, "noise2noise_final_model.keras")
    weights_path = os.path.join(CHECKPOINT_DIR, "final.weights.h5")  # if only has weights file

    generator = load_model_safely(model_path, weights_path)

    if generator is None:
        print("load mode fail")
        generator = build_unet_generator(PATCH_SIZE + (1,), GENERATOR_FILTERS)
    
    input_path = r"C:\Users\HP\GAN_zebrafish\test\t081.tif"
    input_image = tifffile.imread(input_path)
    
    denoised_image = denoise_image(input_image, generator)
    
    output_path = r"C:\Users\HP\GAN_zebrafish\results\denoised_t081.tif"
    tifffile.imwrite(output_path, denoised_image)
    print(f"save denoised image to: {output_path}")
    
    plot_comparison(input_image, denoised_image)

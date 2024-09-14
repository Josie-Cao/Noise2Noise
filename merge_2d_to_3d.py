import os
import numpy as np
import tifffile
from tqdm import tqdm
from config import INPUT_DIR, OUTPUT_DIR, IMAGE_SIZE

def merge_2d_to_3d(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get image size from config file
    num_z_slices, height, width = IMAGE_SIZE

    # Get number of time points
    time_points = len([f for f in os.listdir(input_dir) if f.startswith('aligned_t001_')])

    # Iterate through each time point
    for t in tqdm(range(1, time_points + 1), desc="Processing time points"):
        # Create a list to store all z slices
        volume_slices = []

        # Read all z slices for this time point
        for z in range(1, num_z_slices + 1):
            filename = f"aligned_t{t:03d}_z{z:03d}.tif"
            filepath = os.path.join(input_dir, filename)
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File does not exist: {filepath}")
            
            img = tifffile.imread(filepath)
            volume_slices.append(img)

        # Stack all z slices into a 3D volume
        volume = np.stack(volume_slices, axis=0)

        # Save 3D volume
        output_filename = f"t{t:03d}.tif"
        output_filepath = os.path.join(output_dir, output_filename)
        tifffile.imwrite(output_filepath, volume)

    print("All 3D volumes have been created and saved.")

if __name__ == "__main__":
    merge_2d_to_3d(INPUT_DIR, OUTPUT_DIR)

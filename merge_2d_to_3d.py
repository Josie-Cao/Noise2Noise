import os
import numpy as np
import tifffile
from tqdm import tqdm
from config import INPUT_DIR, OUTPUT_DIR, IMAGE_SIZE

def merge_2d_to_3d(input_dir, output_dir):
    # Merge 2D slices into 3D volumes
    # ... existing code ...

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 从配置文件中获取图像大小
    num_z_slices, height, width = IMAGE_SIZE

    # 获取时间点数量
    time_points = len([f for f in os.listdir(input_dir) if f.startswith('aligned_t001_')])

    # 遍历每个时间点
    for t in tqdm(range(1, time_points + 1), desc="处理时间点"):
        # 创建一个列表来存储所有z切片
        volume_slices = []

        # 读取这个时间点的所有z切片
        for z in range(1, num_z_slices + 1):
            filename = f"aligned_t{t:03d}_z{z:03d}.tif"
            filepath = os.path.join(input_dir, filename)
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"文件不存在: {filepath}")
            
            img = tifffile.imread(filepath)
            volume_slices.append(img)

        # 将所有z切片堆叠成3D体积
        volume = np.stack(volume_slices, axis=0)

        # 保存3D体积
        output_filename = f"t{t:03d}.tif"
        output_filepath = os.path.join(output_dir, output_filename)
        tifffile.imwrite(output_filepath, volume)

    print("所有3D体积已创建并保存。")

if __name__ == "__main__":
    merge_2d_to_3d(INPUT_DIR, OUTPUT_DIR)
import os
import glob
import numpy as np
import pandas as pd
import rasterio


def tif_to_csv(tif_path):
    # 读取tif文件
    with rasterio.open(tif_path) as src:
        # 获取数据
        data = src.read(1)
        # 获取行列数
        rows, cols = data.shape

        # 创建行列索引
        row_indices = np.repeat(np.arange(rows), cols)
        col_indices = np.tile(np.arange(cols), rows)

        # 展平数据
        values = data.flatten()

        # 创建DataFrame
        df = pd.DataFrame({
            'row': row_indices,
            'col': col_indices,
            'value': values
        })

        # 获取文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(tif_path))[0]
        output_dir = os.path.dirname(tif_path)

        # 保存为CSV，保持原始文件名
        output_path = os.path.join(output_dir, f"{base_name}.csv")
        df.to_csv(output_path, index=False)
        print(f"已保存: {output_path}")


def main():
    # 获取D盘下所有的tif文件
    tif_files = glob.glob("D:/hongyouting/data/dnn/auxiliary/025/ST/*.tif")

    if not tif_files:
        print("D盘下没有找到tif文件！")
        return

    for tif_file in tif_files:
        print(f"正在处理: {tif_file}")
        tif_to_csv(tif_file)


if __name__ == "__main__":
    main() 
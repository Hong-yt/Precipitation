import os
import numpy as np
from osgeo import gdal

# 输入原始文件夹路径
input_folder = 'D:/hongyouting/data/dnn/auxiliary/1km/1km/ST'

# 输出裁剪文件夹路径
output_clip_50_folder = 'D:/hongyouting/data/dnn/auxiliary/1km/1km_clip/ST'  # -50~50 裁剪后

# 确保输出目录存在
os.makedirs(output_clip_50_folder, exist_ok=True)

# 获取文件夹中所有tif文件
for file_name in os.listdir(input_folder):
    if file_name.endswith('.tif'):  # 确保只处理.tif文件
        tiff_path = os.path.join(input_folder, file_name)

        # 打开原始栅格文件
        input_raster = gdal.Open(tiff_path)

        # 获取原始像元大小和投影信息
        gt = input_raster.GetGeoTransform()
        projection = input_raster.GetProjection()

        # 定义裁剪的输出路径 (-50~50)
        clipped_tiff_path_50 = os.path.join(output_clip_50_folder, f"1440400_{file_name}")

        # 执行第一次裁剪操作 (-50~50)
        gdal.Translate(clipped_tiff_path_50, input_raster,
                       projWin=[-180, 50, 180, -50],  # [左上角x, 左上角y, 右下角x, 右下角y]
                       outputSRS=projection)


        print(f"文件 {file_name} 裁剪为 -50~50 范围: {clipped_tiff_path_50}")


        # 验证像元大小是否保持一致
        for clip_path, label in [(clipped_tiff_path_50, "-50~50"),]:
            clip_raster = gdal.Open(clip_path)
            clip_gt = clip_raster.GetGeoTransform()
            clip_width = clip_gt[1]
            clip_height = clip_gt[5]
            print(f"{label} 裁剪后像元大小: 宽度={clip_width}, 高度={clip_height}")
            clip_raster = None

        # 关闭数据集
        input_raster = None
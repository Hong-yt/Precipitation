#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import sys
import subprocess
import io
from osgeo import gdal, ogr, osr

# 强制设置控制台输出编码为UTF-8
if sys.platform == 'win32':
    # 修改标准输出和标准错误的编码
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


    # 修改subprocess模块
    def _patch_subprocess():
        orig_popen = subprocess.Popen

        class EncodingFixPopen(orig_popen):
            def __init__(self, *args, **kwargs):
                # 强制设置编码和错误处理
                kwargs['encoding'] = 'utf-8'
                kwargs['errors'] = 'replace'

                # 禁用输出缓冲
                if 'bufsize' not in kwargs:
                    kwargs['bufsize'] = 0

                # 如果是Windows，使用不同的创建标志
                if sys.platform == 'win32' and 'creationflags' not in kwargs:
                    kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW

                super().__init__(*args, **kwargs)

        # 替换原始的Popen
        subprocess.Popen = EncodingFixPopen


    # 应用补丁
    _patch_subprocess()

# 设置GDAL错误处理
gdal.UseExceptions()

# 设置PROJ_LIB环境变量
# os.environ['PROJ_LIB'] = r'G:\data\venv\Lib\site-packages\osgeo\data\proj'

# 输入原始文件夹路径
input_folder = 'D:/hongyouting/data/dnn/auxiliary/1km/1km/ST'

# 输出文件夹路径 - 所有裁剪结果存放在同一个文件夹
output_folder = 'D:/hongyouting/data/dnn/auxiliary/1km/1km_clip/ST'

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹中所有tif文件
for file_name in os.listdir(input_folder):
    if file_name.endswith('.tif'):  # 确保只处理.tif文件
        tiff_path = os.path.join(input_folder, file_name)

        # 文件名（不含扩展名）
        file_base_name = os.path.splitext(file_name)[0]

        # 定义裁剪后的输出文件路径和名称
        clipped_tiff_path_400 = os.path.join(output_folder, f"{file_base_name}_1440400.tif")

        # 如果输出文件已存在，跳过处理
        if os.path.exists(clipped_tiff_path_400):
            print(f"文件 {file_name} 的裁剪版本已存在，跳过处理")
            continue

        try:
            # 打开原始栅格文件
            input_raster = gdal.Open(tiff_path)
            if input_raster is None:
                print(f"无法打开输入文件: {tiff_path}")
                continue

            # 执行裁剪操作 (-50~50)，对应400行
            if not os.path.exists(clipped_tiff_path_400):
                gdal.Warp(clipped_tiff_path_400, input_raster, dstSRS='EPSG:4326', outputBounds=[-180, -50, 180, 50])
                print(f"文件 {file_name} 已裁剪到 -50~50 范围:", clipped_tiff_path_400)

            # 清理资源
            input_raster = None
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {str(e)}")
            continue

print("所有文件处理完成！")
print(f"裁剪结果保存在: {output_folder}")
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

# 输入原始文件夹路径
input_folder = 'D:/hongyouting/data/dnn/auxiliary/1km/1km'

# 输出文件夹路径 - 所有裁剪结果存放在同一个文件夹
output_folder = 'D:/hongyouting/data/dnn/auxiliary/1km/1km_clip'

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹中所有tif文件
for file_name in os.listdir(input_folder):
    if file_name.endswith('.tif'):  # 确保只处理.tif文件
        tiff_path = os.path.join(input_folder, file_name)

        # 文件名（不含扩展名）
        file_base_name = os.path.splitext(file_name)[0]

        # 定义三种裁剪后的输出文件路径和名称
        clipped_tiff_path_400 = os.path.join(output_folder, f"{file_base_name}_1440400.tif")
        clipped_tiff_path_480 = os.path.join(output_folder, f"{file_base_name}_1440480.tif")
        clipped_tiff_path_680 = os.path.join(output_folder, f"{file_base_name}_1440680.tif")

        # 如果所有输出文件都已存在，跳过处理
        if os.path.exists(clipped_tiff_path_400) and os.path.exists(clipped_tiff_path_480) and os.path.exists(
                clipped_tiff_path_680):
            print(f"文件 {file_name} 的所有裁剪版本已存在，跳过处理")
            continue

        # 打开原始栅格文件
        input_raster = gdal.Open(tiff_path)
        if input_raster is None:
            print(f"无法打开输入文件: {tiff_path}")
            continue

        # 执行第一次裁剪操作 (-50~50)，对应400行
        if not os.path.exists(clipped_tiff_path_400):
            gdal.Warp(clipped_tiff_path_400, input_raster, dstSRS='EPSG:4326', outputBounds=[-180, -50, 180, 50])
            print(f"文件 {file_name} 已裁剪到 -50~50 范围:", clipped_tiff_path_400)

        # 执行第二次裁剪操作 (-60~60)，对应480行
        if not os.path.exists(clipped_tiff_path_480):
            gdal.Warp(clipped_tiff_path_480, input_raster, dstSRS='EPSG:4326', outputBounds=[-180, -60, 180, 60])
            print(f"文件 {file_name} 已裁剪到 -60~60 范围:", clipped_tiff_path_480)

        # 执行第三次裁剪操作 (-85~85)，对应680行
        if not os.path.exists(clipped_tiff_path_680):
            gdal.Warp(clipped_tiff_path_680, input_raster, dstSRS='EPSG:4326', outputBounds=[-180, -85, 180, 85])
            print(f"文件 {file_name} 已裁剪到 -85~85 范围:", clipped_tiff_path_680)

        # 清理资源
        input_raster = None

print("所有文件处理完成！")
print(f"裁剪结果保存在: {output_folder}")
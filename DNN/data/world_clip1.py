#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from osgeo import gdal, ogr, osr
import os
import numpy as np
import sys
import subprocess
import io
from osgeo import gdal, ogr, osr
import geopandas as gpd
import rasterio
from rasterio.mask import mask

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

def ensure_dir_exists(dir_path):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"创建目录: {dir_path}")


def mask_tif_with_shp(input_tif, input_shp, output_tif):
    """
    使用shp文件对tif文件进行掩膜裁剪，保持原始tif的行列数、坐标系统和范围不变

    参数:
    input_tif: 输入的tif文件路径
    input_shp: 输入的shp文件路径
    output_tif: 输出的tif文件路径
    """
    try:
        # 读取矢量数据
        gdf = gpd.read_file(input_shp)

        # 确保矢量数据是EPSG:4326坐标系
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")

        # 获取多边形几何体
        geometries = gdf.geometry.values

        # 打开栅格数据
        with rasterio.open(input_tif) as src:
            # 获取原始数据的元数据
            meta = src.meta.copy()

            # 执行掩膜操作，crop=False保持原始边界
            masked_data, masked_transform = mask(src, geometries, crop=False, all_touched=True, nodata=src.nodata)

            # 更新元数据（保持原始元数据不变）
            meta.update({
                "driver": "GTiff",
                "height": masked_data.shape[1],
                "width": masked_data.shape[2],
                "transform": src.transform,  # 保持原始变换
            })

            # 写入新的栅格文件
            with rasterio.open(output_tif, "w", **meta) as dest:
                dest.write(masked_data)

        print(f"成功处理: {os.path.basename(input_tif)}")
        return True
    except Exception as e:
        print(f"处理 {os.path.basename(input_tif)} 时出错: {str(e)}")
        return False


def process_all_tifs(input_dir, output_dir, shp_file):
    """
    处理目录中的所有tif文件

    参数:
    input_dir: 输入tif文件的目录
    output_dir: 输出tif文件的目录
    shp_file: 世界大洲的shp文件路径
    """
    # 确保输出目录存在
    ensure_dir_exists(output_dir)

    # 获取所有tif文件
    tif_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.tif')]

    if not tif_files:
        print(f"在 {input_dir} 中未找到tif文件")
        return

    print(f"找到 {len(tif_files)} 个tif文件待处理")

    # 处理每个tif文件
    successful = 0
    failed = 0

    for tif_file in tif_files:
        input_path = os.path.join(input_dir, tif_file)
        output_path = os.path.join(output_dir, tif_file)

        print(f"正在处理: {tif_file}")
        if mask_tif_with_shp(input_path, shp_file, output_path):
            successful += 1
        else:
            failed += 1

    print(f"\n处理完成! 成功: {successful}, 失败: {failed}")


if __name__ == "__main__":
    # 定义路径
    input_dir = r"D:/hongyouting/data/dnn/auxiliary/1km/1km_clip/ST"
    output_dir = r"D:/hongyouting/data/dnn/auxiliary/1km/1km_world/ST"

    # 世界大洲的shp文件路径 - 请替换为您实际的世界大洲shp文件路径
    # 这里假设您有一个世界大洲的shp文件，如果没有，您需要先获取一个
    continents_shp = r"D:/hongyouting/data/clipshp/世界大洲.shp"  # 请替换为实际的shp文件路径

    # 处理所有tif文件
    process_all_tifs(input_dir, output_dir, continents_shp)
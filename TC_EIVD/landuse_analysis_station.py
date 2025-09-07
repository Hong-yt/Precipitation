# ————————————————————————————————station————————————————————————————————
import pandas as pd
import numpy as np
import rasterio
from tqdm import tqdm

# 1. 加载 station 数据
print("加载 station 数据...")
station_data = pd.read_csv('D:/hongyouting/result/station_statistics/station_imerg_f7.csv')

# 2. 加载站点经纬度信息
print("加载站点经纬度信息...")
station_summary = pd.read_csv('D:/hongyouting/data/station/stations_summary.csv')

# 3. 确保 `station_id` 列的数据类型一致
print("确保 station_id 列的数据类型一致...")
station_data['station_id'] = station_data['station_id'].astype(str)
station_summary['station_id'] = station_summary['station_id'].astype(str)

# 4. 合并 station 数据与站点经纬度信息
print("合并 station 数据与站点经纬度信息...")
station_data = station_data.merge(station_summary[['station_id', 'latitude', 'longitude']],
                                  on='station_id', how='left')

print(f"合并后数据形状: {station_data.shape}")
print(f"包含经纬度信息的站点数: {station_data[['latitude', 'longitude']].notna().all(axis=1).sum()}")

# 5. 加载土地利用类型TIF数据
print("加载土地利用类型TIF数据...")
landuse_tif_path = 'D:/hongyouting/data/landuse/landuse_1440480.tif'  # 请替换为实际的TIF文件路径

with rasterio.open(landuse_tif_path) as src:
    landuse_data = src.read(1)  # 读取第一个波段
    landuse_nodata = src.nodata
    landuse_transform = src.transform  # 获取地理变换参数

    print(f"TIF文件信息: 形状={landuse_data.shape}, NoData值={landuse_nodata}")
    print(f"地理变换参数: {landuse_transform}")

# 6. 定义TIF数据范围和分辨率
print("定义TIF数据网格参数...")
# TIF数据范围: -180到180, -50到50, 网格(1440, 400)
tif_lon_min, tif_lon_max = -180, 180
# tif_lat_min, tif_lat_max = -50, 50
# tif_cols, tif_rows = 1440, 400
tif_lat_min, tif_lat_max = -60, 60
tif_cols, tif_rows = 1440, 480

# 分辨率（假设是0.25度）
resolution = 0.25

print(f"TIF数据网格: {tif_cols} x {tif_rows}")
print(f"TIF数据范围: 经度{tif_lon_min}°到{tif_lon_max}°, 纬度{tif_lat_min}°到{tif_lat_max}°")


# 7. 定义经纬度到网格索引的转换函数
def lonlat_to_grid_index(lon, lat, lon_min, lat_max, resolution):
    """
    将经纬度坐标转换为网格索引
    lon: 经度
    lat: 纬度
    lon_min: 最小经度
    lat_max: 最大纬度
    resolution: 分辨率

    返回: (row_index, col_index)
    """
    # 计算列索引（经度方向）
    col_index = int((lon - lon_min) / resolution)

    # 计算行索引（纬度方向，注意纬度是从上到下递减的）
    row_index = int((lat_max - lat) / resolution)

    return row_index, col_index


# 8. 定义站点土地利用类型匹配函数
def match_station_landuse(lon, lat, landuse_array, nodata_value):
    """
    通过站点经纬度匹配土地利用类型
    lon: 站点经度
    lat: 站点纬度
    landuse_array: 土地利用类型数组
    nodata_value: NoData值
    """
    # 检查经纬度是否有效
    if pd.isna(lon) or pd.isna(lat):
        return np.nan

    # 检查是否在TIF数据范围内
    if not (tif_lon_min <= lon <= tif_lon_max and tif_lat_min <= lat <= tif_lat_max):
        return np.nan

    # 转换为网格索引
    row_idx, col_idx = lonlat_to_grid_index(lon, lat, tif_lon_min, tif_lat_max, resolution)

    # 检查索引是否在数组范围内
    if 0 <= row_idx < tif_rows and 0 <= col_idx < tif_cols:
        value = landuse_array[row_idx, col_idx]
        # 检查是否为NoData值
        if nodata_value is not None and value == nodata_value:
            return np.nan
        else:
            return value
    else:
        return np.nan


# 9. 为每个站点匹配土地利用类型
print("为每个站点匹配土地利用类型...")
tqdm.pandas()  # 激活tqdm进度条

station_data['landuse_type'] = station_data.progress_apply(
    lambda row: match_station_landuse(row['longitude'], row['latitude'], landuse_data, landuse_nodata),
    axis=1
)

# 10. 保存包含所有站点的匹配结果
print("保存所有站点匹配的土地利用类型...")
station_data.to_csv('D:/hongyouting/result/landuse_results/station/imerg_f7/landuse_imerg_f7.csv', index=False)

# 11. 移除无效值（NaN值）
print("移除无效匹配的数据点...")
station_data_valid = station_data.dropna(subset=['landuse_type'])
print(f"有效数据点数量: {len(station_data_valid)} / {len(station_data)}")

# 12. 分析无效数据的分布
station_data_invalid = station_data[station_data['landuse_type'].isna()]
if len(station_data_invalid) > 0:
    print(f"无效数据点分布分析:")
    print(f"  总无效点数: {len(station_data_invalid)}")

    # 分析无效数据的原因
    no_coord = station_data_invalid[station_data_invalid[['latitude', 'longitude']].isna().any(axis=1)]
    out_of_range = station_data_invalid[
        (~station_data_invalid[['latitude', 'longitude']].isna().any(axis=1)) &
        ((station_data_invalid['longitude'] < tif_lon_min) |
         (station_data_invalid['longitude'] > tif_lon_max) |
         (station_data_invalid['latitude'] < tif_lat_min) |
         (station_data_invalid['latitude'] > tif_lat_max))
        ]

    print(f"  缺少经纬度信息: {len(no_coord)} 个")
    print(f"  超出TIF范围: {len(out_of_range)} 个")

    if len(out_of_range) > 0:
        print(f"  超出范围站点的经纬度范围:")
        print(f"    经度: {out_of_range['longitude'].min():.2f}° 到 {out_of_range['longitude'].max():.2f}°")
        print(f"    纬度: {out_of_range['latitude'].min():.2f}° 到 {out_of_range['latitude'].max():.2f}°")

# 13. 根据土地利用类型汇总station结果
print("汇总station结果...")
station_summary_stats = station_data_valid.groupby('landuse_type').agg({
    'rmse': ['mean', 'median', 'std'],
    'r': ['mean', 'median', 'std'],
    'mae': ['mean', 'median', 'std'],
    'norm_rmse': ['mean', 'median', 'std'],
    'norm_mae': ['mean', 'median', 'std'],
    'bias': ['mean', 'median', 'std'],
    'pod': ['mean', 'median', 'std'],
    'far': ['mean', 'median', 'std'],
    'csi': ['mean', 'median', 'std']
})

# 14. 保存汇总结果到CSV文件
print("保存汇总结果到CSV文件...")
station_summary_stats.to_csv('D:/hongyouting/result/landuse_results/station/imerg_f7/landuse_imerg_f7_summary.csv')

# 15. 为每个土地利用类型保存单独的CSV文件
print("为每个土地利用类型保存单独的CSV文件...")
landuse_types = station_data_valid['landuse_type'].unique()
landuse_types = sorted(landuse_types)  # 排序保证顺序一致

for landuse_type in landuse_types:
    # 筛选该土地利用类型的数据
    type_data = station_data_valid[station_data_valid['landuse_type'] == landuse_type]

    # 保存为单独的CSV文件
    filename = f'D:/hongyouting/result/landuse_results/station/imerg_f7/landuse_type_{int(landuse_type)}.csv'
    type_data.to_csv(filename, index=False)

    print(f"  土地利用类型 {int(landuse_type)}: {len(type_data)} 个站点 → {filename}")

print(f"已为 {len(landuse_types)} 个土地利用类型分别保存CSV文件")

# 16. 输出详细统计信息
print("\n=== 统计信息 ===")
print(f"总站点数: {len(station_data)}")
print(f"有效匹配数: {len(station_data_valid)}")
print(f"匹配成功率: {len(station_data_valid) / len(station_data) * 100:.2f}%")
print(f"土地利用类型数量: {station_data_valid['landuse_type'].nunique()}")

if len(station_data_valid) > 0:
    print(f"\n有效站点经纬度范围:")
    print(f"  经度: {station_data_valid['longitude'].min():.2f}° 到 {station_data_valid['longitude'].max():.2f}°")
    print(f"  纬度: {station_data_valid['latitude'].min():.2f}° 到 {station_data_valid['latitude'].max():.2f}°")

print("\n各土地利用类型站点统计:")
landuse_counts = station_data_valid['landuse_type'].value_counts().sort_index()
for landuse_type, count in landuse_counts.items():
    print(f"  类型 {int(landuse_type)}: {count} 个站点")

print("\nStation土地利用类型分析完成，所有结果已保存。")
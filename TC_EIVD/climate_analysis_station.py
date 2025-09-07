import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from tqdm import tqdm
import os

# 1. 加载 station 数据
print("加载 station 数据...")
station_data = pd.read_csv('D:/hongyouting/result/station_statistics/station_cdr.csv')

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

# 5. 加载气候类型 shapefile 数据
print("加载气候类型 shapefile 数据...")
climate_shp = gpd.read_file('D:/hongyouting/data/climate/climatic_type.shp')

# 6. 创建 GeoDataFrame
station_gdf = gpd.GeoDataFrame(station_data, geometry=gpd.points_from_xy(station_data['longitude'],
                                                                         station_data['latitude']),
                               crs=climate_shp.crs)

# 7. 使用空间索引加速匹配
print("构建空间索引...")
climate_sindex = climate_shp.sindex  # 创建空间索引

# 8. 定义加速匹配函数，包含坐标有效性检查
def match_climate_type(point, climate_gdf, sindex):
    possible_matches_index = list(sindex.intersection(point.bounds))
    possible_matches = climate_gdf.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.contains(point)]
    if len(precise_matches) > 0:
        return precise_matches.iloc[0]['CliType']
    else:
        return np.nan

# 9. 为每个 station 数据点匹配气候类型，使用 tqdm 显示进度条
print("为每个 station 数据点匹配气候类型...")
tqdm.pandas()  # 激活 tqdm 进度条
station_gdf['climate_type'] = station_gdf['geometry'].progress_apply(
    lambda point: match_climate_type(point, climate_shp, climate_sindex))

# 10. 保存包含所有点的匹配结果
print("保存所有点匹配的气候类型...")
station_gdf.drop(columns='geometry').to_csv('D:/hongyouting/result/climate_results/STATION/cdr/climate_cdr.csv',
                                            index=False)
# 9. 创建按气候类型分别保存的目录
climate_dir = 'D:/hongyouting/result/climate_results/STATION/cdr'
os.makedirs(climate_dir, exist_ok=True)

# 10. 按气候类型分别保存数据
print("按气候类型分别保存数据...")
# 移除geometry列，准备保存
station_data_clean = station_gdf.drop(columns='geometry')

# 获取所有的气候类型（排除NaN值）
unique_climate_types = station_data_clean['climate_type'].dropna().unique()

print(f"发现 {len(unique_climate_types)} 种气候类型:")
for climate_type in unique_climate_types:
    print(f"  - {climate_type}")

# 为每种气候类型保存单独的CSV文件
for climate_type in tqdm(unique_climate_types, desc="保存各气候类型数据"):
    # 筛选当前气候类型的数据
    climate_data = station_data_clean[station_data_clean['climate_type'] == climate_type]

    # 创建文件名（处理可能的特殊字符）
    safe_climate_name = str(climate_type).replace('/', '_').replace('\\', '_').replace(':', '_')
    filename = f'climate_{safe_climate_name}.csv'
    filepath = os.path.join(climate_dir, filename)

    # 保存数据
    climate_data.to_csv(filepath, index=False)
    print(f"  已保存 {climate_type}: {len(climate_data)} 个数据点 -> {filename}")

# 11. 保存无气候类型匹配的数据
nan_data = station_data_clean[station_data_clean['climate_type'].isna()]
if len(nan_data) > 0:
    nan_filepath = os.path.join(climate_dir, 'climate_no_match.csv')
    nan_data.to_csv(nan_filepath, index=False)
    print(f"  已保存无匹配气候类型数据: {len(nan_data)} 个数据点 -> climate_no_match.csv")


# 11. 根据气候类型汇总 station 结果
print("汇总 station 结果...")
station_summary = station_gdf.groupby('climate_type').agg({
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

# 12. 保存汇总结果到 CSV 文件
print("保存汇总结果到 CSV 文件...")
station_summary.to_csv('D:/hongyouting/result/climate_results/STATION/cdr/climate_cdr_summary.csv')

print("station 计算结果分析完成，所有结果已保存。")
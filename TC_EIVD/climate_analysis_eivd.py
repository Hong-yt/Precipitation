import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from tqdm import tqdm
import os

# 1. 加载eivd数据
print("加载eivd数据...")
eivd_data = pd.read_csv('D:/result/EIVDresults/global2/sm/cdr_f7_EIVD_result.csv')
# eivd_data = pd.read_csv('D:/result/EIVDresults/global2/sm/cdr_e7_EIVD_result.csv')

# 2. 将行列号转换为经纬度
print("转换行列号为经纬度...")
eivd_data['longitude'] = -179.875 + eivd_data['column'] * 0.25
eivd_data['latitude'] = 49.875 - eivd_data['row'] * 0.25

# 3. 加载气候类型shapefile数据
print("加载气候类型shapefile数据...")
climate_shp = gpd.read_file('D:/data/climate/climatic_type.shp')

# 4. 创建 GeoDataFrame
eivd_gdf = gpd.GeoDataFrame(eivd_data, geometry=gpd.points_from_xy(eivd_data['longitude'], eivd_data['latitude']),
                           crs=climate_shp.crs)

# 5. 使用空间索引加速匹配
print("构建空间索引...")
climate_sindex = climate_shp.sindex  # 创建空间索引


# 6. 定义加速匹配函数
def match_climate_type(point, climate_gdf, sindex):
    possible_matches_index = list(sindex.intersection(point.bounds))
    possible_matches = climate_gdf.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.contains(point)]
    if len(precise_matches) > 0:
        return precise_matches.iloc[0]['CliType']
    else:
        return np.nan


# 7. 为每个eivd数据点匹配气候类型，使用 tqdm 显示进度条
print("为每个eivd数据点匹配气候类型...")
tqdm.pandas()  # 激活tqdm进度条
eivd_gdf['climate_type'] = eivd_gdf['geometry'].progress_apply(
    lambda point: match_climate_type(point, climate_shp, climate_sindex))

# 8. 保存包含所有点的匹配结果
print("保存所有点匹配的气候类型...")
eivd_gdf.drop(columns='geometry').to_csv('D:/result/climate_results/EIVD/imerg_f7/climate_imerg_f7.csv',
                                        index=False)
# eivd_gdf.drop(columns='geometry').to_csv('D:/result/climate_results/EIVD/imerg_e7/climate_imerg_e7.csv',
#                                         index=False)

# 9. 创建按气候类型分别保存的目录
climate_dir = 'D:/result/climate_results/EIVD/imerg_f7'
os.makedirs(climate_dir, exist_ok=True)

# 10. 按气候类型分别保存数据
print("按气候类型分别保存数据...")
# 移除geometry列，准备保存
eivd_data_clean = eivd_gdf.drop(columns='geometry')

# 获取所有的气候类型（排除NaN值）
unique_climate_types = eivd_data_clean['climate_type'].dropna().unique()

print(f"发现 {len(unique_climate_types)} 种气候类型:")
for climate_type in unique_climate_types:
    print(f"  - {climate_type}")

# 为每种气候类型保存单独的CSV文件
for climate_type in tqdm(unique_climate_types, desc="保存各气候类型数据"):
    # 筛选当前气候类型的数据
    climate_data = eivd_data_clean[eivd_data_clean['climate_type'] == climate_type]

    # 创建文件名（处理可能的特殊字符）
    safe_climate_name = str(climate_type).replace('/', '_').replace('\\', '_').replace(':', '_')
    filename = f'climate_{safe_climate_name}.csv'
    filepath = os.path.join(climate_dir, filename)

    # 保存数据
    climate_data.to_csv(filepath, index=False)
    print(f"  已保存 {climate_type}: {len(climate_data)} 个数据点 -> {filename}")

# 11. 保存无气候类型匹配的数据
nan_data = eivd_data_clean[eivd_data_clean['climate_type'].isna()]
if len(nan_data) > 0:
    nan_filepath = os.path.join(climate_dir, 'climate_no_match.csv')
    nan_data.to_csv(nan_filepath, index=False)
    print(f"  已保存无匹配气候类型数据: {len(nan_data)} 个数据点 -> climate_no_match.csv")

# 12. 根据气候类型汇总eivd结果
print("汇总eivd结果...")
eivd_summary = eivd_gdf.groupby('climate_type').agg({
        'SM2RAIN_stderr': ['mean', 'median', 'std'],
        'SM2RAIN_rho': ['mean', 'median', 'std'],
        'SM2RAIN_snr': ['mean', 'median', 'std'],
        'SM2RAIN_fmse': ['mean', 'median', 'std'],
        'sate1_stderr': ['mean', 'median', 'std'],
        'sate1_rho': ['mean', 'median', 'std'],
        'sate1_snr': ['mean', 'median', 'std'],
        'sate1_fmse': ['mean', 'median', 'std'],
        'sate2_stderr': ['mean', 'median', 'std'],
        'sate2_rho': ['mean', 'median', 'std'],
        'sate2_snr': ['mean', 'median', 'std'],
        'sate2_fmse': ['mean', 'median', 'std'],
})

# 13. 保存汇总结果到CSV文件
print("保存汇总结果到CSV文件...")
eivd_summary.to_csv('D:/result/climate_results/EIVD/imerg_f7/climate_imerg_f7_summary.csv')
# eivd_summary.to_csv('D:/result/climate_results/EIVD/imerg_e7/climate_imerg_e7_summary.csv')

# 14. 生成数据统计报告
print("\n=== 数据统计报告 ===")
print(f"总数据点数: {len(eivd_data_clean)}")
print(f"成功匹配气候类型的数据点: {len(eivd_data_clean.dropna(subset=['climate_type']))}")
print(f"未匹配气候类型的数据点: {len(nan_data)}")
print(f"气候类型数量: {len(unique_climate_types)}")

# 各气候类型数据点统计
print("\n各气候类型数据点统计:")
climate_counts = eivd_data_clean['climate_type'].value_counts()
for climate_type, count in climate_counts.items():
    print(f"  {climate_type}: {count} 个数据点")

print(f"\neivd效果分析完成，所有结果已保存。")
print(f"各气候类型数据已保存到: {climate_dir}")
print("文件列表:")
for filename in os.listdir(climate_dir):
    if filename.endswith('.csv'):
        print(f"  - {filename}")
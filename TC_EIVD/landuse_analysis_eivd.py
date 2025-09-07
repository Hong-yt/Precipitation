import pandas as pd
import numpy as np
import rasterio
from tqdm import tqdm

# 1. 加载eivd数据
print("加载eivd数据...")
eivd_data = pd.read_csv('D:/hongyouting/result/EIVDresults/global2/sm/nrt_e7_EIVD_result.csv')

# 2. 加载土地利用类型TIF数据
print("加载土地利用类型TIF数据...")
landuse_tif_path = 'D:/hongyouting/data/landuse/landuse_1440400.tif'  # 请替换为实际的TIF文件路径

with rasterio.open(landuse_tif_path) as src:
    landuse_data = src.read(1)  # 读取第一个波段
    landuse_nodata = src.nodata

    print(f"TIF文件信息: 形状={landuse_data.shape}, NoData值={landuse_nodata}")

# 3. 定义数据范围和分辨率
print("定义数据网格参数...")
# eivd数据范围: -180到180, -90到90, 网格(1440, 720)
eivd_lon_min, eivd_lon_max = -180, 180
# eivd_lat_min, eivd_lat_max = -60, 60
# eivd_cols, eivd_rows = 1440, 480
eivd_lat_min, eivd_lat_max = -50, 50
eivd_cols, eivd_rows = 1440, 400

# TIF数据范围: -180到180, -85到85, 网格(1440, 680)
tif_lon_min, tif_lon_max = -180, 180
# tif_lat_min, tif_lat_max = -60, 60
# tif_cols, tif_rows = 1440, 480
tif_lat_min, tif_lat_max = -50, 50
tif_cols, tif_rows = 1440, 400

# 分辨率（假设都是0.25度）
resolution = 0.25

print(f"eivd数据网格: {eivd_cols} x {eivd_rows}")
print(f"TIF数据网格: {tif_cols} x {tif_rows}")
# print(f"匹配规则: eivd行号20-699 → TIF行号0-679 (去掉上下各20行)")


# 4. 定义网格匹配函数
def match_landuse_grid(row, column, landuse_array, nodata_value):
    """
    通过网格行列号直接匹配土地利用类型
    i: eivd数据的行号 (0-719)
    j: eivd数据的列号 (0-1439)

    两个数据集的差异：
    - eivd: 720行，纬度范围 90°到-90° (行号0-719)
    - TIF: 680行，纬度范围 85°到-85° (行号0-679)
    - 差异：上面5°(20行) + 下面5°(20行) = 40行
    """
    # 简单的行号转换：eivd去掉上面20行，下面20行
    # eivd的行号20-699对应TIF的行号0-679
    # tif_i = int(i) - 20
    tif_row = int(row)
    tif_column = int(column)  # 列号直接对应（经度范围相同）

    # 检查是否在TIF数据范围内
    if 0 <= tif_row < tif_rows and 0 <= tif_column < tif_cols:
        value = landuse_array[tif_row, tif_column]
        # 检查是否为NoData值
        if nodata_value is not None and value == nodata_value:
            return np.nan
        else:
            return value
    else:
        return np.nan


# 5. 为每个eivd数据点匹配土地利用类型
print("为每个eivd数据点匹配土地利用类型...")
tqdm.pandas()  # 激活tqdm进度条

eivd_data['landuse_type'] = eivd_data.progress_apply(
    lambda row: match_landuse_grid(row['row'], row['column'], landuse_data, landuse_nodata),
    axis=1
)

# 6. 计算经纬度（用于验证和保存）
print("计算经纬度坐标...")
# 网格规则：行列号从0开始，从左到右，从上到下
# j=0对应经度-179.875°，j递增经度递增
# i=0对应纬度89.875°，i递增纬度递减
eivd_data['longitude'] = -179.875 + eivd_data['column'] * 0.25
eivd_data['latitude'] = 49.875 - eivd_data['row'] * 0.25

# 7. 保存包含所有点的匹配结果
print("保存所有点匹配的土地利用类型...")
eivd_data.to_csv('D:/hongyouting/result/landuse_results/EIVD/gsmap_nrt/landuse_gsmap_nrt.csv', index=False)

# 8. 移除无效值（NaN值）
print("移除无效匹配的数据点...")
eivd_data_valid = eivd_data.dropna(subset=['landuse_type'])
print(f"有效数据点数量: {len(eivd_data_valid)} / {len(eivd_data)}")

# 9. 分析无效数据的分布
eivd_data_invalid = eivd_data[eivd_data['landuse_type'].isna()]
if len(eivd_data_invalid) > 0:
    print(f"无效数据点分布分析:")
    print(f"  总无效点数: {len(eivd_data_invalid)}")

    # 分析行号分布
    invalid_rows = eivd_data_invalid['i'].value_counts().sort_index()
    print(f"  主要分布在行号: {invalid_rows.index.min()}-{invalid_rows.index.max()}")

    # # 统计超出TIF范围的点（行号<20或>699）
    # out_of_range_north = eivd_data_invalid[eivd_data_invalid['i'] < 20]
    # out_of_range_south = eivd_data_invalid[eivd_data_invalid['i'] > 699]
    # print(f"  北极地区(行号0-19): {len(out_of_range_north)} 个")
    # print(f"  南极地区(行号700-719): {len(out_of_range_south)} 个")

# 10. 根据土地利用类型汇总eivd结果
print("汇总eivd结果...")
eivd_summary = eivd_data_valid.groupby('landuse_type').agg({
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

# 11. 保存汇总结果到CSV文件
print("保存汇总结果到CSV文件...")
eivd_summary.to_csv('D:/hongyouting/result/landuse_results/EIVD/gsmap_nrt/landuse_gsmap_nrt_summary.csv')

# 12. 为每个土地利用类型保存单独的CSV文件
print("为每个土地利用类型保存单独的CSV文件...")
landuse_types = eivd_data_valid['landuse_type'].unique()
landuse_types = sorted(landuse_types)  # 排序保证顺序一致

for landuse_type in landuse_types:
    # 筛选该土地利用类型的数据
    type_data = eivd_data_valid[eivd_data_valid['landuse_type'] == landuse_type]

    # 保存为单独的CSV文件
    filename = f'D:/hongyouting/result/landuse_results/EIVD/gsmap_nrt/landuse_type_{int(landuse_type)}.csv'
    type_data.to_csv(filename, index=False)

    print(f"  土地利用类型 {int(landuse_type)}: {len(type_data)} 个数据点 → {filename}")

print(f"已为 {len(landuse_types)} 个土地利用类型分别保存CSV文件")

# 13. 输出详细统计信息
print("\n=== 统计信息 ===")
print(f"总数据点数: {len(eivd_data)}")
print(f"有效匹配数: {len(eivd_data_valid)}")
print(f"匹配成功率: {len(eivd_data_valid) / len(eivd_data) * 100:.2f}%")
print(f"土地利用类型数量: {eivd_data_valid['landuse_type'].nunique()}")

print(f"\n有效数据行号范围: {eivd_data_valid['row'].min()} 到 {eivd_data_valid['row'].max()}")
print(f"对应纬度范围: {eivd_data_valid['latitude'].max():.2f}° 到 {eivd_data_valid['latitude'].min():.2f}°")
# print(f"TIF数据覆盖范围: 85° 到 -85°")

print("\n各土地利用类型数据点统计:")
landuse_counts = eivd_data_valid['landuse_type'].value_counts().sort_index()
for landuse_type, count in landuse_counts.items():
    print(f"  类型 {int(landuse_type)}: {count} 个数据点")

print("\neivd土地利用类型分析完成，所有结果已保存。")
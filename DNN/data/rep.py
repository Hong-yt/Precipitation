import math

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds, calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
from scipy.stats import rankdata
import os
from tqdm import tqdm


def calculate_spatial_representativeness(auxiliary_025_path, auxiliary_1km_path, station_csv_path, output_csv_path,
                                         detailed_output_path=None, debug=False, force_resolution=True):
    """
    计算站点auxiliary的空间代表性

    参数:
    -----------
    auxiliary_025_path : str
        0.25°分辨率auxiliary栅格文件路径
    auxiliary_1km_path : str
        1km分辨率auxiliary栅格文件路径
    station_csv_path : str
        站点CSV文件路径，包含station_id、latitude和longitude三列
    output_csv_path : str
        输出CSV文件路径，用于保存计算得到的代表性值
    detailed_output_path : str, optional
        详细输出CSV文件路径，包含更多计算过程的中间结果
    debug : bool, optional
        是否打印调试信息，默认为False
    force_resolution : bool, optional
        是否强制使用固定分辨率，默认为True
    """
    # 读取站点数据
    stations = pd.read_csv(station_csv_path)
    print(f"从{station_csv_path}加载了{len(stations)}个站点")

    # 读取auxiliary栅格数据
    with rasterio.open(auxiliary_025_path) as auxiliary_025_src:
        # 检查auxiliary是否覆盖预期范围
        bounds = auxiliary_025_src.bounds
        # print(f"0.25°auxiliary边界: {bounds}")

        # 计算0.25°网格单元的大小
        cell_width = 0.25  # 度
        cell_height = 0.25  # 度

    # 读取1km栅格数据并检查分辨率
    with rasterio.open(auxiliary_1km_path) as auxiliary_1km_src:
        # 检查1km auxiliary是否覆盖预期范围
        bounds_1km = auxiliary_1km_src.bounds
        print(f"1km auxiliary边界: {bounds_1km}")

        # 检查数据类型和nodata值
        print(f"数据类型: {auxiliary_1km_src.dtypes}")
        print(f"Nodata值: {auxiliary_1km_src.nodata}")

        # 检查实际分辨率
        original_res_x, original_res_y = auxiliary_1km_src.res
        print(f"1km auxiliary原始分辨率: {original_res_x}°, {original_res_y}°")
        print(f"换算为公里: 约{original_res_x * 111}km, {original_res_y * 111}km (赤道附近)")

        # 对于1km分辨率数据，强制设定理想的分辨率
        if force_resolution:
            # 对于1km分辨率，0.25°应该对应约30×30像素（考虑到地球曲率和像素形状）
            force_pixels_per_grid = 30  # 每个0.25°网格的像素数
            res_x = cell_width / force_pixels_per_grid
            res_y = cell_height / force_pixels_per_grid
            print(f"强制使用分辨率: {res_x}°, {res_y}°")
            print(f"换算为公里: 约{res_x * 111}km, {res_y * 111}km (赤道附近)")
        else:
            res_x, res_y = original_res_x, original_res_y

        # 计算0.25°理论应包含的像素数
        theoretical_pixels_x = int(cell_width / res_x)
        theoretical_pixels_y = int(cell_height / res_y)
        theoretical_pixels = theoretical_pixels_x * theoretical_pixels_y
        print(f"理论上0.25°网格应包含约{theoretical_pixels}个像素 ({theoretical_pixels_x}x{theoretical_pixels_y})")

        # 如果分辨率与1km相差太大且没有强制使用固定分辨率，给出警告
        if not force_resolution and (abs(res_x * 111 - 1) > 0.2 or abs(res_y * 111 - 1) > 0.2):
            print(f"警告: 栅格数据分辨率可能不是1km! 请检查数据源。")

    # 初始化结果数据框
    results = pd.DataFrame({
        'station_id': stations['station_id'],
        'latitude': stations['latitude'],
        'longitude': stations['longitude'],
        'auxiliary_representativeness': np.nan
    })

    # 初始化详细结果数据框
    detailed_results = pd.DataFrame({
        'station_id': stations['station_id'],
        'latitude': stations['latitude'],
        'longitude': stations['longitude'],
        'grid_cell_x': np.nan,
        'grid_cell_y': np.nan,
        'cell_min_x': np.nan,
        'cell_min_y': np.nan,
        'cell_max_x': np.nan,
        'cell_max_y': np.nan,
        'window_width': np.nan,
        'window_height': np.nan,
        'station_auxiliary_value': np.nan,
        'station_category': np.nan,
        'total_valid_cells': np.nan,
        'expected_valid_cells': np.nan,
        'same_category_count': np.nan,
        'auxiliary_representativeness': np.nan,
        'min_auxiliary_in_grid': np.nan,
        'max_auxiliary_in_grid': np.nan,
        'mean_auxiliary_in_grid': np.nan,
        'std_auxiliary_in_grid': np.nan,
        'processing_error': ''  # 新增错误信息字段
    })

    # 创建一个字典来跟踪每个1km网格单元中的站点
    grid_stations = {}

    # 设置浮点数比较容差
    FLOAT_TOLERANCE = 1e-6  # 根据数据精度可能需要调整

    # 处理每个站点
    for idx, station in tqdm(stations.iterrows(), total=len(stations), desc="处理站点"):
        station_id = station['station_id']
        lat = station['latitude']
        lon = station['longitude']

        # 检查经纬度是否为NaN
        if pd.isna(lat) or pd.isna(lon):
            print(f"站点{station_id}的经纬度包含NaN值。跳过处理。")
            detailed_results.loc[idx, 'processing_error'] = '经纬度含NaN值'
            continue

        # 跳过超出有效范围的站点
        if lat <= -50 or lat >= 50 or lon <= -180 or lon >= 180:
            print(f"站点{station_id}位于{lat}, {lon}，超出有效范围。跳过处理。")
            detailed_results.loc[idx, 'processing_error'] = '经纬度超出有效范围'
            continue

        # 计算包含此站点的0.25°网格单元
        grid_cell_x = 719 + math.ceil(lon / cell_height)
        grid_cell_y = 199 - math.floor(lat / cell_width)

        # 计算这个0.25°网格单元的边界
        cell_min_x = -180 + grid_cell_x * cell_width
        cell_max_x = cell_min_x + cell_width
        cell_min_y = 50 - (grid_cell_y + 1) * cell_height
        cell_max_y = cell_min_y + cell_height

        # 保存网格信息到详细结果
        detailed_results.loc[idx, 'grid_cell_x'] = grid_cell_x
        detailed_results.loc[idx, 'grid_cell_y'] = grid_cell_y
        detailed_results.loc[idx, 'cell_min_x'] = cell_min_x
        detailed_results.loc[idx, 'cell_min_y'] = cell_min_y
        detailed_results.loc[idx, 'cell_max_x'] = cell_max_x
        detailed_results.loc[idx, 'cell_max_y'] = cell_max_y

        # 提取该0.25°网格单元内所有1km网格单元的auxiliary值
        try:
            with rasterio.open(auxiliary_1km_path) as auxiliary_1km_src:
                # 如果强制使用固定分辨率，直接计算窗口大小
                if force_resolution:
                    # 计算0.25°网格单元的中心点
                    center_x = (cell_min_x + cell_max_x) / 2
                    center_y = (cell_min_y + cell_max_y) / 2

                    # 计算中心点的像素坐标
                    center_row, center_col = auxiliary_1km_src.index(center_x, center_y)

                    # 计算窗口左上角坐标（从中心点向外扩展）
                    half_width = theoretical_pixels_x // 2
                    half_height = theoretical_pixels_y // 2

                    # 获取左上角坐标
                    col_min = center_col - half_width
                    row_min = center_row - half_height

                    # 确保窗口在栅格范围内
                    if col_min < 0:
                        col_min = 0
                    if row_min < 0:
                        row_min = 0

                    # 设置固定窗口大小
                    window_width = theoretical_pixels_x
                    window_height = theoretical_pixels_y

                    # 确保窗口不超出栅格范围
                    if col_min + window_width > auxiliary_1km_src.width:
                        window_width = auxiliary_1km_src.width - col_min
                    if row_min + window_height > auxiliary_1km_src.height:
                        window_height = auxiliary_1km_src.height - row_min
                else:
                    # 获取栅格的实际分辨率
                    res_x, res_y = auxiliary_1km_src.res

                    # 计算0.25°网格单元在1km栅格中的像素坐标
                    row_min, col_min = auxiliary_1km_src.index(cell_min_x, cell_max_y)  # 左上角
                    row_max, col_max = auxiliary_1km_src.index(cell_max_x, cell_min_y)  # 右下角

                    # 确保窗口有效
                    if row_min > row_max:
                        row_min, row_max = row_max, row_min
                    if col_min > col_max:
                        col_min, col_max = col_max, col_min

                    # 计算正确的窗口大小
                    expected_width = min(int(cell_width / res_x) + 1, col_max - col_min)
                    expected_height = min(int(cell_height / res_y) + 1, row_max - row_min)

                    # 限制窗口大小，防止异常值
                    MAX_WINDOW_SIZE = max(theoretical_pixels_x, theoretical_pixels_y) * 2  # 设置合理的上限
                    window_width = min(MAX_WINDOW_SIZE, col_max - col_min)
                    window_height = min(MAX_WINDOW_SIZE, row_max - row_min)

                # 保存窗口大小到详细结果
                detailed_results.loc[idx, 'window_width'] = window_width
                detailed_results.loc[idx, 'window_height'] = window_height
                detailed_results.loc[idx, 'expected_valid_cells'] = theoretical_pixels

                # if debug:
                #     print(f"\n站点{station_id}:")
                #     print(f"  经纬度边界: {cell_min_x}, {cell_min_y}, {cell_max_x}, {cell_max_y}")
                #     if not force_resolution:
                #         print(f"  像素坐标: 行({row_min}-{row_max}), 列({col_min}-{col_max})")
                #         print(f"  原始窗口大小: {col_max - col_min} x {row_max - row_min}")
                #     print(f"  计算的窗口大小: {window_width} x {window_height}")
                #     if not force_resolution:
                #         print(f"  理论期望的窗口大小: {expected_width} x {expected_height}")
                #     print(f"  理论期望的像素数: {theoretical_pixels}")
                #     print(f"  是否使用固定分辨率: {'是' if force_resolution else '否'}")

                # 提取0.25°网格单元的auxiliary值，使用调整后的窗口大小
                window = Window(col_min, row_min, window_width, window_height)
                auxiliary_values = auxiliary_1km_src.read(1, window=window)

                # 检查是否获取到数据
                if auxiliary_values.size == 0:
                    print(f"站点{station_id}：未能获取到辅助数据（窗口大小为0）。跳过处理。")
                    detailed_results.loc[idx, 'processing_error'] = '窗口大小为0'
                    continue

                # 计算此0.25°网格单元中的1km网格单元数量
                if auxiliary_1km_src.nodata is not None:
                    total_cells = (auxiliary_values != auxiliary_1km_src.nodata).sum()
                else:
                    total_cells = auxiliary_values.size

                # if debug:
                #     print(f"  窗口内总单元格数: {auxiliary_values.size}")
                #     print(f"  有效单元格数: {total_cells}")
                #     if auxiliary_1km_src.nodata is not None:
                #         print(f"  Nodata值: {auxiliary_1km_src.nodata}")
                #         print(f"  Nodata单元格数: {(auxiliary_values == auxiliary_1km_src.nodata).sum()}")

                # 获取站点位置的auxiliary值
                row, col = auxiliary_1km_src.index(lon, lat)
                try:
                    station_auxiliary_value = auxiliary_1km_src.read(1, window=Window(col, row, 1, 1))[0, 0]
                    detailed_results.loc[idx, 'station_auxiliary_value'] = station_auxiliary_value

                    if debug:
                        print(f"  站点辅助值: {station_auxiliary_value}")
                except IndexError:
                    print(f"站点{station_id}位于{lat}, {lon}，超出1km auxiliary范围。跳过处理。")
                    detailed_results.loc[idx, 'processing_error'] = '站点超出auxiliary范围'
                    continue

                # 如果站点或太多单元格没有数据，则跳过
                if (
                        auxiliary_1km_src.nodata is not None and station_auxiliary_value == auxiliary_1km_src.nodata) or total_cells < 10:
                    print(
                        f"站点{station_id}没有有效的auxiliary数据或有效单元格太少(total_cells={total_cells})。跳过处理。")
                    detailed_results.loc[idx, 'processing_error'] = f'有效单元格太少({total_cells})'
                    continue

                # 移除无数据值
                if auxiliary_1km_src.nodata is not None:
                    valid_mask = auxiliary_values != auxiliary_1km_src.nodata
                    valid_auxiliary_values = auxiliary_values[valid_mask]
                else:
                    valid_auxiliary_values = auxiliary_values.flatten()
                    valid_mask = np.ones_like(auxiliary_values, dtype=bool)

                # 检查有效值数组是否为空
                if len(valid_auxiliary_values) == 0:
                    print(f"站点{station_id}：有效值数组为空。跳过处理。")
                    detailed_results.loc[idx, 'processing_error'] = '有效值数组为空'
                    continue

                # 检查实际有效单元格数量与理论值的差异
                actual_valid_cells = len(valid_auxiliary_values)
                if debug:
                    print(f"  有效辅助值数组长度: {actual_valid_cells}")

                if abs(actual_valid_cells - theoretical_pixels) > theoretical_pixels * 0.5 and theoretical_pixels > 0:
                    print(
                        f"警告: 站点{station_id}的有效单元格数量({actual_valid_cells})与理论值({theoretical_pixels})相差较大!")

                # 保存网格auxiliary统计信息
                if len(valid_auxiliary_values) > 0:
                    detailed_results.loc[idx, 'min_auxiliary_in_grid'] = np.min(valid_auxiliary_values)
                    detailed_results.loc[idx, 'max_auxiliary_in_grid'] = np.max(valid_auxiliary_values)
                    detailed_results.loc[idx, 'mean_auxiliary_in_grid'] = np.mean(valid_auxiliary_values)
                    detailed_results.loc[idx, 'std_auxiliary_in_grid'] = np.std(valid_auxiliary_values)
                    detailed_results.loc[idx, 'total_valid_cells'] = len(valid_auxiliary_values)

                    # if debug:
                    #     print(f"  最小值: {np.min(valid_auxiliary_values)}")
                    #     print(f"  最大值: {np.max(valid_auxiliary_values)}")
                    #     print(f"  平均值: {np.mean(valid_auxiliary_values)}")
                    #     print(f"  标准差: {np.std(valid_auxiliary_values)}")

                # 基于四分位数将auxiliary值分为20个类别
                if len(valid_auxiliary_values) > 0:
                    # 计算排名(从1开始)并转换为20个类别
                    ranks = rankdata(valid_auxiliary_values, method='average')
                    categories = np.ceil(ranks * 20 / len(valid_auxiliary_values)).astype(int)

                    # 创建一个类别映射，用于处理多站点情况
                    category_map = {}
                    for i, val in enumerate(valid_auxiliary_values):
                        category_map[val] = categories[i]

                    # 使用近似匹配找出站点auxiliary值所属的类别
                    # 替换精确匹配为近似匹配
                    # station_index = np.where(valid_auxiliary_values == station_auxiliary_value)[0]
                    station_index = \
                    np.where(np.abs(valid_auxiliary_values - station_auxiliary_value) < FLOAT_TOLERANCE)[0]

                    if len(station_index) > 0:
                        station_category = categories[station_index[0]]
                        detailed_results.loc[idx, 'station_category'] = station_category

                        # 统计有多少单元格具有相同类别
                        same_category_count = np.sum(categories == station_category)
                        detailed_results.loc[idx, 'same_category_count'] = same_category_count

                        # 计算代表性
                        representativeness = same_category_count / len(valid_auxiliary_values)

                        # 存储结果
                        results.loc[idx, 'auxiliary_representativeness'] = representativeness
                        detailed_results.loc[idx, 'auxiliary_representativeness'] = representativeness

                        print(f"站点{station_id}：auxiliary代表性 = {representativeness:.4f}")

                        # 处理1km网格中的多站点情况
                        # 使用(row, col)作为网格单元的唯一标识
                        grid_key = (row, col)
                        if grid_key not in grid_stations:
                            grid_stations[grid_key] = []

                        grid_stations[grid_key].append({
                            'station_id': station_id,
                            'latitude': lat,
                            'longitude': lon,
                            'auxiliary_value': station_auxiliary_value,
                            'category': station_category,
                            'representativeness': representativeness
                        })
                    else:
                        # 如果找不到完全匹配的值，尝试找到最接近的值
                        closest_index = np.argmin(np.abs(valid_auxiliary_values - station_auxiliary_value))
                        closest_value = valid_auxiliary_values[closest_index]
                        difference = abs(closest_value - station_auxiliary_value)

                        print(f"在网格单元数据中找不到站点{station_id}的auxiliary值({station_auxiliary_value})。")
                        print(f"最接近的值: {closest_value}，差异: {difference}")

                        if difference < FLOAT_TOLERANCE * 10:  # 使用较宽松的容差再次尝试
                            print(f"使用最接近的值({closest_value})计算代表性。")

                            station_category = categories[closest_index]
                            detailed_results.loc[idx, 'station_category'] = station_category

                            # 统计有多少单元格具有相同类别
                            same_category_count = np.sum(categories == station_category)
                            detailed_results.loc[idx, 'same_category_count'] = same_category_count

                            # 计算代表性
                            representativeness = same_category_count / len(valid_auxiliary_values)

                            # 存储结果
                            results.loc[idx, 'auxiliary_representativeness'] = representativeness
                            detailed_results.loc[idx, 'auxiliary_representativeness'] = representativeness

                            print(f"站点{station_id}(使用近似值)：auxiliary代表性 = {representativeness:.4f}")
                        else:
                            detailed_results.loc[idx, 'processing_error'] = f'找不到匹配的辅助值(差异={difference})'
                else:
                    print(f"站点{station_id}的网格单元中没有找到有效的auxiliary值。")
                    detailed_results.loc[idx, 'processing_error'] = '网格单元中没有有效的辅助值'
        except Exception as e:
            print(f"处理站点{station_id}时发生错误: {str(e)}")
            detailed_results.loc[idx, 'processing_error'] = f'处理错误: {str(e)}'
            continue

    # 保存结果
    results.to_csv(output_csv_path, index=False)
    print(f"结果已保存到{output_csv_path}")

    # 保存详细结果
    if detailed_output_path:
        detailed_results.to_csv(detailed_output_path, index=False)
        print(f"详细结果已保存到{detailed_output_path}")

    # 创建多站点网格报告
    multi_station_grids = {k: v for k, v in grid_stations.items() if len(v) > 1}
    if multi_station_grids:
        multi_station_report_path = os.path.splitext(output_csv_path)[0] + "_multi_station_report.csv"
        multi_station_rows = []

        for grid_key, stations_list in multi_station_grids.items():
            row, col = grid_key
            for i, station_info in enumerate(stations_list):
                multi_station_rows.append({
                    'grid_row': row,
                    'grid_col': col,
                    'grid_station_count': len(stations_list),
                    'station_id': station_info['station_id'],
                    'latitude': station_info['latitude'],
                    'longitude': station_info['longitude'],
                    'auxiliary_value': station_info['auxiliary_value'],
                    'category': station_info['category'],
                    'representativeness': station_info['representativeness']
                })

        if multi_station_rows:
            multi_station_df = pd.DataFrame(multi_station_rows)
            multi_station_df.to_csv(multi_station_report_path, index=False)
            print(f"多站点网格报告已保存到{multi_station_report_path}")
            print(f"发现{len(multi_station_grids)}个包含多个站点的1km网格单元")

    # 统计处理结果
    processed_count = len(stations)
    success_count = results['auxiliary_representativeness'].notna().sum()
    error_count = detailed_results['processing_error'].notna().sum()

    print("\n处理统计:")
    print(f"总站点数: {processed_count}")
    print(f"成功计算代表性的站点数: {success_count} ({success_count / processed_count * 100:.1f}%)")
    print(f"处理失败的站点数: {error_count} ({error_count / processed_count * 100:.1f}%)")

    # 如果有错误，显示主要错误类型
    if error_count > 0:
        error_types = detailed_results['processing_error'].value_counts()
        print("\n主要错误类型:")
        for error_type, count in error_types.items():
            if pd.notna(error_type) and error_type != '':
                print(f"  - {error_type}: {count}个站点")

    return results, detailed_results


# 示例用法
if __name__ == "__main__":
    auxiliary_025_path = "D:/hongyouting/data/dnn/auxiliary/025/DEM/DEM_1440400.tif"  # 0.25°DEM文件
    auxiliary_1km_path = "D:/hongyouting/data/dnn/auxiliary/1km/1km_world/1440400_DEM.tif"  # 1km DEM文件
    station_csv_path = "D:/hongyouting/data/station/stations_summary.csv"  # 站点数据
    output_csv_path = "D:/hongyouting/data/dnn/rep/dem_representativeness_results.csv"  # 输出文件
    detailed_output_path = "D:/hongyouting/data/dnn/rep/dem_representativeness_detailed.csv"  # 详细输出文件

    # 设置debug=True可以查看更多调试信息，force_resolution=True强制使用固定分辨率
    results, detailed_results = calculate_spatial_representativeness(
        auxiliary_025_path, auxiliary_1km_path, station_csv_path, output_csv_path, detailed_output_path,
        debug=True, force_resolution=True
    )
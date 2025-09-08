import math

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds, calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
from scipy.stats import rankdata
import os
from tqdm import tqdm


def calculate_landuse_representativeness(landuse_025_path, landuse_1km_path, station_csv_path, output_csv_path,
                                         detailed_output_path=None, debug=False, force_resolution=True,
                                         igbp_class_names=None):
    """
    计算站点土地利用类型(landuse)的空间代表性，基于IGBP分类体系

    参数:
    -----------
    landuse_025_path : str
        0.25°分辨率landuse栅格文件路径
    landuse_1km_path : str
        1km分辨率landuse栅格文件路径
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
    igbp_class_names : dict, optional
        IGBP类别编码到类名的映射字典，如果为None，则使用默认值
    """
    # IGBP分类方案对应的类别名称（默认值）
    if igbp_class_names is None:
        igbp_class_names = {
            1: "常绿针叶林",
            2: "常绿阔叶林",
            3: "落叶针叶林",
            4: "落叶阔叶林",
            5: "混交林",
            6: "封闭灌丛",
            7: "开放灌丛",
            8: "木本稀树草原",
            9: "稀树草原",
            10: "草原",
            11: "永久湿地",
            12: "农田",
            13: "城市和建筑用地",
            14: "农田/自然植被镶嵌体",
            15: "雪和冰",
            16: "裸地或稀疏植被",
            17: "水体",
            255: "填充值/无数据"
        }

    # 读取站点数据
    stations = pd.read_csv(station_csv_path)
    print(f"从{station_csv_path}加载了{len(stations)}个站点")

    # 读取landuse栅格数据
    with rasterio.open(landuse_025_path) as landuse_025_src:
        # 检查landuse是否覆盖预期范围
        bounds = landuse_025_src.bounds
        print(f"0.25°landuse边界: {bounds}")

        # 计算0.25°网格单元的大小
        cell_width = 0.25  # 度
        cell_height = 0.25  # 度

    # 读取1km栅格数据并检查分辨率
    with rasterio.open(landuse_1km_path) as landuse_1km_src:
        # 检查1km landuse是否覆盖预期范围
        bounds_1km = landuse_1km_src.bounds
        print(f"1km landuse边界: {bounds_1km}")

        # 检查实际分辨率
        original_res_x, original_res_y = landuse_1km_src.res
        print(f"1km landuse原始分辨率: {original_res_x}°, {original_res_y}°")
        print(f"换算为公里: 约{original_res_x * 111}km, {original_res_y * 111}km (赤道附近)")

        # 获取实际的nodata值
        nodata_value = landuse_1km_src.nodata
        print(f"Land use数据的无数据值: {nodata_value}")

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
        'landuse_class': np.nan,
        'landuse_class_name': "",
        'landuse_representativeness': np.nan
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
        'station_landuse_class': np.nan,
        'station_landuse_class_name': "",
        'total_valid_cells': np.nan,
        'expected_valid_cells': np.nan,
        'same_class_count': np.nan,
        'landuse_representativeness': np.nan,
        'class_distribution': ""
    })

    # 创建一个字典来跟踪每个1km网格单元中的站点
    grid_stations = {}

    # 处理每个站点
    for idx, station in tqdm(stations.iterrows(), total=len(stations), desc="处理站点"):
        station_id = station['station_id']
        lat = station['latitude']
        lon = station['longitude']

        # 检查经纬度是否为NaN
        if pd.isna(lat) or pd.isna(lon):
            print(f"站点{station_id}的经纬度包含NaN值。跳过处理。")
            continue

        # 跳过超出有效范围的站点
        if lat <= -50 or lat >= 50 or lon <= -180 or lon >= 180:
            print(f"站点{station_id}位于{lat}, {lon}，超出有效范围。跳过处理。")
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

        # 提取该0.25°网格单元内所有1km网格单元的landuse值
        with rasterio.open(landuse_1km_path) as landuse_1km_src:
            # 如果强制使用固定分辨率，直接计算窗口大小
            if force_resolution:
                # 计算0.25°网格单元的中心点
                center_x = (cell_min_x + cell_max_x) / 2
                center_y = (cell_min_y + cell_max_y) / 2

                # 计算中心点的像素坐标
                center_row, center_col = landuse_1km_src.index(center_x, center_y)

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
                if col_min + window_width > landuse_1km_src.width:
                    window_width = landuse_1km_src.width - col_min
                if row_min + window_height > landuse_1km_src.height:
                    window_height = landuse_1km_src.height - row_min
            else:
                # 获取栅格的实际分辨率
                res_x, res_y = landuse_1km_src.res

                # 计算0.25°网格单元在1km栅格中的像素坐标
                row_min, col_min = landuse_1km_src.index(cell_min_x, cell_max_y)  # 左上角
                row_max, col_max = landuse_1km_src.index(cell_max_x, cell_min_y)  # 右下角

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

            if debug:
                print(f"\n站点{station_id}:")
                print(f"  经纬度边界: {cell_min_x}, {cell_min_y}, {cell_max_x}, {cell_max_y}")
                if not force_resolution:
                    print(f"  像素坐标: 行({row_min}-{row_max}), 列({col_min}-{col_max})")
                    print(f"  原始窗口大小: {col_max - col_min} x {row_max - row_min}")
                print(f"  计算的窗口大小: {window_width} x {window_height}")
                if not force_resolution:
                    print(f"  理论期望的窗口大小: {expected_width} x {expected_height}")
                print(f"  理论期望的像素数: {theoretical_pixels}")
                print(f"  是否使用固定分辨率: {'是' if force_resolution else '否'}")

            # 提取0.25°网格单元的landuse值，使用调整后的窗口大小
            window = Window(col_min, row_min, window_width, window_height)
            landuse_values = landuse_1km_src.read(1, window=window)

            # 计算此0.25°网格单元中的1km网格单元数量
            if landuse_1km_src.nodata is not None:
                total_cells = (landuse_values != landuse_1km_src.nodata).sum()
            else:
                total_cells = landuse_values.size

            if debug:
                print(f"  窗口内总单元格数: {landuse_values.size}")
                print(f"  有效单元格数: {total_cells}")

            # 获取站点位置的landuse值
            row, col = landuse_1km_src.index(lon, lat)
            try:
                station_landuse_class = landuse_1km_src.read(1, window=Window(col, row, 1, 1))[0, 0]
                detailed_results.loc[idx, 'station_landuse_class'] = station_landuse_class

                # 获取土地利用类别名称
                if station_landuse_class in igbp_class_names:
                    station_landuse_class_name = igbp_class_names[station_landuse_class]
                    detailed_results.loc[idx, 'station_landuse_class_name'] = station_landuse_class_name
                    results.loc[idx, 'landuse_class'] = station_landuse_class
                    results.loc[idx, 'landuse_class_name'] = station_landuse_class_name
                else:
                    print(f"警告: 站点{station_id}的土地利用类别{station_landuse_class}在IGBP分类中不存在")
                    station_landuse_class_name = f"未知类别({station_landuse_class})"
                    detailed_results.loc[idx, 'station_landuse_class_name'] = station_landuse_class_name
                    results.loc[idx, 'landuse_class'] = station_landuse_class
                    results.loc[idx, 'landuse_class_name'] = station_landuse_class_name

            except IndexError:
                print(f"站点{station_id}位于{lat}, {lon}，超出1km landuse范围。跳过处理。")
                continue

            # 如果站点或太多单元格没有数据，则跳过
            if (
                    landuse_1km_src.nodata is not None and station_landuse_class == landuse_1km_src.nodata) or total_cells < 10:
                print(f"站点{station_id}没有有效的landuse数据或有效单元格太少。跳过处理。")
                continue

            # 移除无数据值
            if landuse_1km_src.nodata is not None:
                valid_mask = landuse_values != landuse_1km_src.nodata
                valid_landuse_values = landuse_values[valid_mask]
            else:
                valid_landuse_values = landuse_values.flatten()
                valid_mask = np.ones_like(landuse_values, dtype=bool)

            # 检查实际有效单元格数量与理论值的差异
            actual_valid_cells = len(valid_landuse_values)
            if abs(actual_valid_cells - theoretical_pixels) > theoretical_pixels * 0.5 and theoretical_pixels > 0:
                print(
                    f"警告: 站点{station_id}的有效单元格数量({actual_valid_cells})与理论值({theoretical_pixels})相差较大!")

            detailed_results.loc[idx, 'total_valid_cells'] = actual_valid_cells

            # 土地利用类型直接使用IGBP分类，不需要使用四分位数分类
            if len(valid_landuse_values) > 0:
                # 统计各类别的数量
                unique_classes, class_counts = np.unique(valid_landuse_values, return_counts=True)

                # 创建类别分布字符串，保存到详细结果
                class_distribution = {}
                for i, cls in enumerate(unique_classes):
                    cls_name = igbp_class_names.get(cls, f"未知类别({cls})")
                    percentage = (class_counts[i] / len(valid_landuse_values)) * 100
                    class_distribution[cls_name] = f"{percentage:.2f}%"

                detailed_results.loc[idx, 'class_distribution'] = str(class_distribution)

                # 计算与站点相同土地利用类型的网格单元数量
                same_class_count = np.sum(valid_landuse_values == station_landuse_class)
                detailed_results.loc[idx, 'same_class_count'] = same_class_count

                # 计算代表性
                representativeness = same_class_count / len(valid_landuse_values)

                # 存储结果
                results.loc[idx, 'landuse_representativeness'] = representativeness
                detailed_results.loc[idx, 'landuse_representativeness'] = representativeness

                print(f"站点{station_id}：土地利用类型 = {station_landuse_class_name}，代表性 = {representativeness:.4f}")

                # 处理1km网格中的多站点情况
                # 使用(row, col)作为网格单元的唯一标识
                grid_key = (row, col)
                if grid_key not in grid_stations:
                    grid_stations[grid_key] = []

                grid_stations[grid_key].append({
                    'station_id': station_id,
                    'latitude': lat,
                    'longitude': lon,
                    'landuse_class': station_landuse_class,
                    'landuse_class_name': station_landuse_class_name,
                    'representativeness': representativeness
                })
            else:
                print(f"站点{station_id}的网格单元中没有找到有效的landuse值。")

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
                    'landuse_class': station_info['landuse_class'],
                    'landuse_class_name': station_info['landuse_class_name'],
                    'representativeness': station_info['representativeness']
                })

        if multi_station_rows:
            multi_station_df = pd.DataFrame(multi_station_rows)
            multi_station_df.to_csv(multi_station_report_path, index=False)
            print(f"多站点网格报告已保存到{multi_station_report_path}")
            print(f"发现{len(multi_station_grids)}个包含多个站点的1km网格单元")

    return results, detailed_results


# 示例用法
if __name__ == "__main__":
    # IGBP土地利用分类
    igbp_class_names = {
        1: "常绿针叶林",
        2: "常绿阔叶林",
        3: "落叶针叶林",
        4: "落叶阔叶林",
        5: "混交林",
        6: "封闭灌丛",
        7: "开放灌丛",
        8: "木本稀树草原",
        9: "稀树草原",
        10: "草原",
        11: "永久湿地",
        12: "农田",
        13: "城市和建筑用地",
        14: "农田/自然植被镶嵌体",
        15: "雪和冰",
        16: "裸地或稀疏植被",
        17: "水体",
        255: "填充值/无数据"
    }

    landuse_025_path = "D:/hongyouting/some_preprocess/dnn/auxiliary/025/landuse/landuse_1440400.tif"  # 0.25°landuse文件
    landuse_1km_path = "D:/hongyouting/some_preprocess/dnn/auxiliary/1km/1km_world/1440400_landuse.tif"  # 1km landuse文件
    station_csv_path = "D:/hongyouting/some_preprocess/station/stations_summary.csv"  # 站点数据
    output_csv_path = "D:/hongyouting/some_preprocess/dnn/rep/landuse_representativeness_results.csv"  # 输出文件
    detailed_output_path = "D:/hongyouting/some_preprocess/dnn/rep/landuse_representativeness_detailed.csv"  # 详细输出文件

    # 设置debug=True可以查看更多调试信息，force_resolution=True强制使用固定分辨率
    results, detailed_results = calculate_landuse_representativeness(
        landuse_025_path, landuse_1km_path, station_csv_path, output_csv_path, detailed_output_path,
        debug=True, force_resolution=True, igbp_class_names=igbp_class_names
    )
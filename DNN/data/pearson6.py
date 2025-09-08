import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import os


def calculate_correlation_pair(data1, data2, output_prefix, output_dir):
    """
    计算两个数据集之间的相关系数R

    参数:
    data1: 第一个数据集数组
    data2: 第二个数据集数组
    output_prefix: 输出文件前缀
    output_dir: 输出目录路径

    返回:
    mean_r: 所有网格的平均相关系数R
    """
    # 确认数据形状
    print(f"计算 {output_prefix} 的相关系数...")
    print(f"数据集1形状: {data1.shape}")
    print(f"数据集2形状: {data2.shape}")

    if data1.shape != data2.shape:
        raise ValueError("两个数据集的形状必须相同")

    # 从形状中提取维度
    n_samples, n_rows, n_cols = data1.shape

    # 初始化结果数组
    r_values = np.zeros((n_rows, n_cols))
    p_values = np.zeros((n_rows, n_cols))

    print("计算每个网格的相关系数...")
    # 对每个网格计算相关系数
    for i in range(n_rows):
        if i % 50 == 0:
            print(f"处理行 {i}/{n_rows}")
        for j in range(n_cols):
            # 提取特定网格的时间序列
            series1 = data1[:, i, j]
            series2 = data2[:, i, j]

            # 检查是否有效数据（非NaN）
            valid_indices = ~(np.isnan(series1) | np.isnan(series2))
            valid_series1 = series1[valid_indices]
            valid_series2 = series2[valid_indices]

            # 仅当有足够的有效数据点时计算相关系数
            if len(valid_series1) > 1:
                r, p = pearsonr(valid_series1, valid_series2)
                r_values[i, j] = r
                p_values[i, j] = p
            else:
                r_values[i, j] = np.nan
                p_values[i, j] = np.nan

    # 创建包含网格行列号的DataFrame
    print(f"保存 {output_prefix} 的相关系数矩阵到CSV（包含网格行列号）...")

    # 创建用于保存结果的列表
    results = []

    # 填充结果列表，包含行号、列号、R值和P值
    for i in range(n_rows):
        for j in range(n_cols):
            results.append({
                'row': i,
                'col': j,
                'r_value': r_values[i, j],
                'p_value': p_values[i, j]
            })

    # 创建DataFrame
    results_df = pd.DataFrame(results)

    # 保存为CSV
    results_csv_path = os.path.join(output_dir, f"{output_prefix}.csv")
    results_df.to_csv(results_csv_path, index=False)

    # 为了向后兼容，同时保存原始矩阵格式
    r_df = pd.DataFrame(r_values)
    p_df = pd.DataFrame(p_values)

    r_csv_path = os.path.join(output_dir, f"{output_prefix}_r.csv")
    p_csv_path = os.path.join(output_dir, f"{output_prefix}_p.csv")

    r_df.to_csv(r_csv_path, index=True)  # 保存行索引
    p_df.to_csv(p_csv_path, index=True)  # 保存行索引

    # 计算平均相关系数（忽略NaN值）
    mean_r = np.nanmean(r_values)

    # 保存平均相关系数
    summary_path = os.path.join(output_dir, f"{output_prefix}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"平均相关系数R: {mean_r}\n")
        f.write(f"有效网格数: {np.sum(~np.isnan(r_values))}\n")
        f.write(f"总网格数: {n_rows * n_cols}\n")

    print(f"{output_prefix} 平均相关系数R: {mean_r}")

    return mean_r, r_values, p_values


def calculate_multiple_correlations(dataset1_path, dataset2_path, dataset3_path, dataset1_name, output_dir):
    """
    计算多组数据集之间的相关系数

    参数:
    dataset1_path: 第一个数据集的npz文件路径
    dataset2_path: 第二个数据集的npz文件路径
    dataset3_path: 第三个数据集的npz文件路径
    dataset1_name: dataset1 的自定义名称（例如 "GSMaP"）

    output_dir: 输出目录路径
    """

    dataset2_name = "ERA5"
    dataset3_name = "SM2RAIN"

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载npz数据集
    print("加载npz数据集...")
    data1_npz = np.load(dataset1_path)
    data2_npz = np.load(dataset2_path)
    data3_npz = np.load(dataset3_path)

    # 获取npz文件中的数组
    data1_keys = list(data1_npz.keys())
    data2_keys = list(data2_npz.keys())
    data3_keys = list(data3_npz.keys())

    print(f"数据集1中的数组键: {data1_keys}")
    print(f"数据集2中的数组键: {data2_keys}")
    print(f"数据集3中的数组键: {data3_keys}")

    # 获取实际数据
    data1 = data1_npz[data1_keys[0]]
    data2 = data2_npz[data2_keys[0]]
    data3 = data3_npz[data3_keys[0]]

    # 确认所有数据集形状相同
    print(f"数据集1形状: {data1.shape}")
    print(f"数据集2形状: {data2.shape}")
    print(f"数据集3形状: {data3.shape}")

    if data1.shape != data2.shape or data1.shape != data3.shape:
        raise ValueError("所有数据集的形状必须相同")

    # 计算 dataset1 与 ERA5 之间的相关系数
    mean_r_1_2, _, _ = calculate_correlation_pair(
        data1, data2, f"{dataset1_name}_{dataset2_name}", output_dir
    )

    # 计算 dataset1 与 SM2RAIN 之间的相关系数
    mean_r_1_3, _, _ = calculate_correlation_pair(
        data1, data3, f"{dataset1_name}_{dataset3_name}", output_dir
    )

    # 保存总体比较结果
    overall_summary_path = os.path.join(output_dir, "CHIRPS.txt")
    with open(overall_summary_path, "w") as f:
        f.write(f"{dataset1_name} 与 {dataset2_name} 平均相关系数R: {mean_r_1_2}\n")
        f.write(f"{dataset1_name} 与 {dataset3_name} 平均相关系数R: {mean_r_1_3}\n")
        f.write(f"相关系数差异({dataset1_name}_{dataset2_name} - {dataset1_name}_{dataset3_name}): {mean_r_1_2 - mean_r_1_3}\n")

    print("计算完成!")
    print(f"{dataset1_name} 与 {dataset2_name} 平均相关系数R: {mean_r_1_2}")
    print(f"{dataset1_name} 与 {dataset3_name} 平均相关系数R: {mean_r_1_3}")
    print(f"所有结果已保存到 {output_dir}")

    return {
        f'mean_r_{dataset1_name}_{dataset2_name}': mean_r_1_2,
        f'mean_r_{dataset1_name}_{dataset3_name}': mean_r_1_3
    }


if __name__ == "__main__":
    dataset1_path = "D:/hongyouting/data/0.25_clip_50/4930/chirps_precipitation.npz"
    dataset2_path = "D:/hongyouting/data/0.25_clip_50/4930/era5_precipitation.npz"
    dataset3_path = "D:/hongyouting/data/0.25_clip_50/4930/sm2rain_precipitation.npz"
    output_dir = "D:/hongyouting/result/data_r"

    dataset1_name = "CHIRPS"  # 你可以修改这里的名称
    results = calculate_multiple_correlations(dataset1_path, dataset2_path, dataset3_path, dataset1_name, output_dir)

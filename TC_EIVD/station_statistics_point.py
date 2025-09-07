import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import time
import tkinter as tk
from tkinter import messagebox

def nonnan(x, y):
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x = x[mask]
    y = y[mask]
    return x, y


def RMSE(x, y):
    x, y = nonnan(x, y)
    # print(f"RMSE - x: {x}, y: {y}")
    if len(x) != len(y):
        raise ValueError('length of x is not equal to y')
    elif len(x) == 0:
        return np.nan
    else:
        return (((x - y) ** 2).sum() / len(x)) ** 0.5


def R(x, y):
    x, y = nonnan(x, y)
    # print(f"R - x: {x}, y: {y}")
    if len(x) != len(y):
        raise ValueError('length of x is not equal to y')
    elif len(x) <= 1:
        return np.nan
    elif np.std(x) == 0 or np.std(y) == 0:
        return np.nan  # 如果x或y的标准差为0，返回NaN(常量)
    else:
        return pearsonr(x, y)[0]


def normRMSE(x, y):
    x, y = nonnan(x, y)
    # print(f"normRMSE - x: {x}, y: {y}")
    if len(x) == 0:
        return np.nan
    elif x.max() - x.mean() == 0:
        return np.nan
    else:
        return RMSE(x, y) / (x.max() - x.mean())


def MAE(x, y):
    x, y = nonnan(x, y)
    # print(f"MAE - x: {x}, y: {y}")
    if len(x) != len(y):
        raise ValueError('length of x is not equal to y')
    elif len(x) == 0:
        return np.nan
    else:
        return np.abs(x - y).sum() / len(x)


def normMAE(x, y):
    x, y = nonnan(x, y)
    # print(f"normMAE - x: {x}, y: {y}")
    if len(x) == 0:
        return np.nan
    elif x.max() - x.mean() == 0:
        return np.nan
    else:
        return MAE(x, y) / (x.max() - x.mean())


def totalVolumeRatio(x, y):
    x, y = nonnan(x, y)
    # print(f"totalVolumeRatio - x: {x}, y: {y}")
    if len(x) == 0 or y.sum() == 0:
        return np.nan
    else:
        return x.sum() / y.sum()


def POD(x, y, threshold=0.2):
    x, y = nonnan(x, y)
    # print(f"POD - x: {x}, y: {y}")
    a = (x >= threshold) & (y >= threshold)
    b = (x < threshold) & (y >= threshold)
    if len(a) == 0 or (a.sum() + b.sum()) == 0:
        return np.nan
    else:
        return a.sum() / (a.sum() + b.sum())


def FAR(x, y, threshold=0.2):
    x, y = nonnan(x, y)
    # print(f"FAR - x: {x}, y: {y}")
    c = (x >= threshold) & (y < threshold)
    a = (x >= threshold) & (y >= threshold)
    if len(a) == 0 or (a.sum() + c.sum()) == 0:
        return np.nan
    else:
        return c.sum() / (a.sum() + c.sum())


def CSI(x, y, threshold=0.2):
    x, y = nonnan(x, y)
    # print(f"CSI - x: {x}, y: {y}")
    a = (x >= threshold) & (y >= threshold)
    b = (x < threshold) & (y >= threshold)
    c = (x >= threshold) & (y < threshold)
    if len(a) == 0 or (a.sum() + b.sum() + c.sum()) == 0:
        return np.nan
    else:
        return a.sum() / (a.sum() + b.sum() + c.sum())


def Sum(x):
    y = np.zeros(len(x))
    x, y = nonnan(x, y)
    # print(f"Sum - x: {x}, y: {y}")
    if len(x) == 0:
        return np.nan
    else:
        return x.sum()

# def parse_date(date_str):
#     # 统一日期格式：尝试多种常见格式
#     for fmt in ('%Y/%m/%d', '%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y'):
#         try:
#             return pd.to_datetime(date_str, format=fmt)
#         except ValueError:
#             continue
#     return pd.NaT  # 如果所有格式都失败，则返回 NaT

def read_station_data(folder_path):
    station_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path, header=None, encoding='utf-8-sig', sep=',', skip_blank_lines=True)

            # 统一日期格式并过滤掉日期范围
            data[1] = pd.to_datetime(data[1], format='%Y-%m-%d', errors='coerce')
            data = data[data[1] <= pd.Timestamp('2020-06-30')]
            data = data[data[1] >= pd.Timestamp('2007-01-01')]

            # 过滤掉第五列值为0、2539.746或空值的行
            data = data[(data[4] != 0) & (data[4] != 2539.746) & (~data[4].isna())]

            # 检查过滤后的数据是否为空
            if data.empty:
                print(f"Skipped empty data after filtering from {filename}.")
                continue

            if len(data) >= 363:  # 检查数据行数是否大于等于363
                station_data.append(data)
            print(f"{filename}:\n", data.head())  # 打印解析后的数据以检查

    if station_data:
        station_data = pd.concat(station_data, ignore_index=True)
    else:
        station_data = pd.DataFrame()  # 如果没有有效数据文件，返回空的DataFrame
    return station_data


# def read_station_data(folder_path):
#     station_data = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.csv'):
#             file_path = os.path.join(folder_path, filename)
#             try:
#                 if os.path.getsize(file_path) > 0:  # Check if the file is not empty
#                     data = pd.read_csv(file_path, header=None)
#                 else:
#                     print(f"Skipped empty file: {filename}")
#                     continue
#
#                 # Filter dates greater than 2020/6/30
#                 data[1] = pd.to_datetime(data[1], format='%Y/%m/%d', errors='coerce')
#                 data = data[data[1] <= pd.Timestamp('2020-06-30')]
#
#                 if len(data) >= 363:  # Check if data has at least 363 rows
#                     station_data.append(data)
#             except pd.errors.EmptyDataError:
#                 print(f"Skipped empty file: {filename}")
#                 continue
#
#     if station_data:
#         station_data = pd.concat(station_data, ignore_index=True)
#     else:
#         station_data = pd.DataFrame()  # Return an empty DataFrame if no valid data files are found
#     return station_data

def read_satellite_data(file_path):
    data = np.load(file_path)['array']
    return data


def calculate_statistics(station_data, satellite_data, lat_range, lon_range, output_file):
    stats = {
        'station_id': [],
        'rmse': [],
        'r': [],
        'mae': [],
        'norm_rmse': [],
        'norm_mae': [],
        'bias': [],
        'pod': [],
        'far': [],
        'csi': [],
    }
    # print(station_data.head())

    station_data[4] = station_data[4].replace(2539.746, np.nan)
    station_ids = station_data[0].unique()

    for station_id in station_ids:
        station_subset = station_data[station_data[0] == station_id]

        station_precips = []
        satellite_precips = []

        for index, row in station_subset.iterrows():
            station_date = pd.to_datetime(row[1], format='%Y-%m-%d', errors='coerce')

            if pd.isna(station_date):
                continue

            station_lat = row[2]
            station_lon = row[3]
            station_precip = row[4]

            if np.isnan(station_precip):
                continue

            # if station_lat < -90 or station_lat > 90:  # 纬度范围限制
            #     continue
            if station_lat < -60 or station_lat > 60:  # 纬度范围限制
                continue
            # if station_lat < -50 or station_lat > 50:  # 纬度范围限制
            #     continue

            day_of_year = (station_date - pd.Timestamp("2007-01-01")).days  # 注意起始时间

            # if day_of_year < 0:
            #     continue
            #
            # # Adjust index for missing dates
            # if station_date > pd.Timestamp("2007-02-27"):
            #     day_of_year -= 3
            #
            # if day_of_year >= satellite_data.shape[0]:
            #     continue

            if day_of_year < 0 or day_of_year >= satellite_data.shape[0]:
                print(f"Date out of range for station {station_id} on day {day_of_year}")
                continue

            satellite_precip = satellite_data[day_of_year, :, :]

            if np.isnan(station_lat) or np.isnan(station_lon):
                print(f"跳过站点 {station_id}，纬度或经度缺失.")
                continue

            # lat_idx = np.abs(lat_range - station_lat).argmin()
            # lon_idx = np.abs(lon_range - station_lon).argmin()
            # 计算索引
            # lat_idx = int((90 - station_lat) / 0.25)  # 纬度索引
            # lon_idx = int((station_lon + 180) / 0.25)  # 经度索引
            lat_idx = int((60 - station_lat) / 0.25)  # 纬度索引
            lon_idx = int((station_lon + 180) / 0.25)  # 经度索引
            # lat_idx = int((50 - station_lat) / 0.25)  # 纬度索引
            # lon_idx = int((station_lon + 180) / 0.25)  # 经度索引

            # print(f"Latitude Index: {lat_idx}, Longitude Index: {lon_idx}")

            if lat_idx < 0 or lat_idx >= satellite_precip.shape[0] or lon_idx < 0 or lon_idx >= satellite_precip.shape[
                1]:
                # print(f"Latitude or longitude out of range for station {station_id}")
                continue

            satellite_precip_value = satellite_precip[lat_idx, lon_idx]

            if np.isnan(satellite_precip_value):  # 检查网格数据集中的值是否为空值
                continue

            station_precips.append(station_precip)
            satellite_precips.append(satellite_precip_value)

        if len(station_precips) == 0 or len(satellite_precips) == 0:
            continue

        station_precips = np.array(station_precips)
        satellite_precips = np.array(satellite_precips)

        rmse_val = RMSE(station_precips, satellite_precips)
        mae_val = MAE(station_precips, satellite_precips)
        norm_rmse_val = normRMSE(station_precips, satellite_precips)
        norm_mae_val = normMAE(station_precips, satellite_precips)
        r_val = R(station_precips, satellite_precips)
        bias_val = totalVolumeRatio(station_precips, satellite_precips)
        pod_val = POD(station_precips, satellite_precips)
        far_val = FAR(station_precips, satellite_precips)
        csi_val = CSI(station_precips, satellite_precips)

        stats['station_id'].append(station_id)
        stats['rmse'].append(rmse_val)
        stats['r'].append(r_val)
        stats['mae'].append(mae_val)
        stats['norm_rmse'].append(norm_rmse_val)
        stats['norm_mae'].append(norm_mae_val)
        stats['bias'].append(bias_val)
        stats['pod'].append(pod_val)
        stats['far'].append(far_val)
        stats['csi'].append(csi_val)

        # # 在弹窗中显示站点数据
        # show_station_data(station_id, station_precips, satellite_precips, r_val)

        df = pd.DataFrame(stats)
        df.to_csv(output_file, index=False, mode='a', header=not os.path.exists(output_file))  # 追加模式保存结果

        # 清空stats字典，以便于下一个站点的计算
        stats = {
            'station_id': [],
            'rmse': [],
            'r': [],
            'mae': [],
            'norm_rmse': [],
            'norm_mae': [],
            'bias': [],
            'pod': [],
            'far': [],
            'csi': [],
        }

        print(f"Station {station_id}: RMSE={rmse_val}, R={r_val}, MAE={mae_val}, normRMSE={norm_rmse_val}, normMAE={norm_mae_val}, "
              f"Bias={bias_val}, POD={pod_val}, FAR={far_val}, CSI={csi_val}")

    return stats


if __name__ == '__main__':
    start_time = time.time()  # 记录开始时间

    # station_folder = 'D:/hongyouting/data/station/GSOD'
    # satellite_file = 'D:/hongyouting/data/0.25_clip/imerg_f7_4930_precipitation.npz'
    # output_file = 'D:/hongyouting/result/station_statistics/1/station_imerg_f7.csv'

    station_folder = 'D:/hongyouting/data/station/GSOD4_nan'
    satellite_file = 'D:/hongyouting/data/0.25_clip_60/cmorph_precipitation.npz'
    output_file = 'D:/hongyouting/result/station_statistics/1/station_cmorph.csv'

    # station_folder = 'D:/hongyouting/data/station/GSOD'
    # satellite_file = 'D:/hongyouting/data/0.25_clip_60/cdr_precipitation.npy'
    # output_file = 'D:/hongyouting/result/station_statistics/station_cdr.csv'

    # station_folder = 'D:/hongyouting/data/station/GSOD'
    # satellite_file = 'D:/hongyouting/data/0.25_clip_50/chirps_precipitation.npy'
    # output_file = 'D:/hongyouting/result/station_statistics/station_chirps.csv'


    station_data = read_station_data(station_folder)
    satellite_data = read_satellite_data(satellite_file)

    # 假设 lat_range 和 lon_range 是已知的经纬度范围
    # lat_range = np.linspace(90, -90, 720)
    lat_range = np.linspace(60, -60, 480)
    # lat_range = np.linspace(50, -50, 400)
    lon_range = np.linspace(-180, 180, 1440)

    statistics = calculate_statistics(station_data, satellite_data, lat_range, lon_range, output_file)

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算程序运行时间

    print(f"程序运行时间: {elapsed_time:.2f} 秒")

import math
import os
import numpy as np
import pandas as pd
from numpy.ma.core import append
from scipy.stats import pearsonr
import time

#
# def nonnan(x, y):
#     mask = (~np.isnan(x)) & (~np.isnan(y))
#     x = x[mask]
#     y = y[mask]
#     return x, y


def nonnan(x, y):
    # 过滤掉包含 NaN 的值
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x = x[mask]
    y = y[mask]

    # 调试输出：打印过滤后数组的形状
    # print(f"nonnan - x: {x.shape}, y: {y.shape}")

    return x, y

# def nonnan(x, y):
#     # 将 x 和 y 转换为 numpy 数组（如果它们不是的话）
#     x = np.asarray(x)
#     y = np.asarray(y)
#
#     # 创建一个布尔数组，标记有效的 (非 NaN) 数据
#     mask = ~np.isnan(x) & ~np.isnan(y)
#
#     # 只保留有效数据
#     x = x[mask]
#     y = y[mask]
#
#     return x, y

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
    print(f"R - x: {x}, y: {y}")
    # print(f"R - x: {x.shape}, y: {y.shape}")

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
    a = (x >= threshold) & (y >= threshold)
    c = (x >= threshold) & (y < threshold)
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



def read_station_data(folder_path):
    print(f"---加载站点---")
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
            data = data[(data[4] != 2539.746) & (~data[4].isna())]

            # 检查过滤后的数据是否为空
            if data.empty:
                print(f"Skipped empty data after filtering from {filename}.")
                continue

            if len(data) >= 10:  # 检查数据行数是否大于等于363
                station_data.append(data)
            # print(f"{filename}:\n", data.head())  # 打印解析后的数据以检查
            # print(f"{filename}")  # 打印解析后的数据以检查）

    if station_data:
        station_data = pd.concat(station_data, ignore_index=True)
    else:
        station_data = pd.DataFrame()  # 如果没有有效数据文件，返回空的DataFrame

    return station_data


def read_satellite_data(file_path):
    print(f"---加载卫星数据---")
    data = np.load(file_path)['array']
    return data

# target_lat_idx = 49
# target_lon_idx = 752

def calculate_statistics(station_data, satellite_data, output_file):
    print(f"---开始计算---")

    # 初始化一个列表来存储每个网格的统计结果
    stats = {
        'lat_idx': [],
        'lon_idx': [],
        'r': [],
        'rmse': [],
        'mae': [],
        'norm_rmse': [],
        'norm_mae': [],
        'bias': [],
        'pod': [],
        'far': [],
        'csi': [],
    }

    # 检查站点数据是否为空
    if station_data.empty:
        print("站点数据为空！")
        return
    # 输出加载的部分站点数据，检查是否正确加载
    # print(f"---加载的站点数据预览---")
    # print(station_data.head())  # 打印站点数据的前几行，检查格式

    # 输出加载的部分卫星数据，检查是否正确加载
    # print(f"---加载的卫星数据预览---")
    # print(f"卫星数据的形状: {satellite_data.shape}")  # 打印卫星数据的形状
    # print(f"卫星数据样本（49，752 数据）: {satellite_data[0, 49, 752]}")  # 打印卫星数据的一个样本值

    # if satellite_data.empty:
    #     print("卫星数据为空！")
    #     return
    # 用一个字典来存储每个网格的站点降水值，键为(lat_idx, lon_idx)，值为对应日期下的降水值列表
    grid_data = {}
    # processed_stations = set()  # 存储已处理的站点编号

    print(f"---站点相关计算中---")
    print(f"satellite shap: (row: {satellite_data.shape[1]}, col: {satellite_data.shape[2]}):")

    # 遍历站点数据，计算每个站点的网格行列号
    for _, row in station_data.iterrows():

        station_lat = row[2]
        station_lon = row[3]
        station_precip = row[4]
        station_ids = row[0]  # 获取站点ID
        station_date = pd.to_datetime(row[1], format='%Y/%m/%d', errors='coerce')  # 确保日期正确格式化

        if pd.isna(station_date):
            print(f"跳过站点，日期缺失.")
            continue

        # if station_lat < -90 or station_lat > 90:  # 纬度范围限制
        #     continue
        if station_lat < -60 or station_lat > 60:  # 纬度范围限制
            continue
        # if station_lat < -50 or station_lat > 50:  # 纬度范围限制
        #     continue

        if np.isnan(station_lat) or np.isnan(station_lon):
            print(f"跳过站点，纬度或经度缺失.")
            continue

        # 计算该站点所在网格的行列号（lat_idx, lon_idx）
        # if station_ids not in processed_stations:
        #     processed_stations.add(station_ids)
        # 计算网格索引
        if station_lat >= 0:
            lat_idx = 239 - math.floor(station_lat / 0.25)
        else:
            lat_idx = 239 - math.floor(station_lat / 0.25)

        if station_lon >= 0:
            lon_idx = 719 + math.ceil(station_lon / 0.25)
        else:
            lon_idx = 719 + math.ceil(station_lon / 0.25)

        # print(f"station{station_ids}: (row: {lat_idx}, col: {lon_idx}):")

        if lat_idx < 0 or lat_idx >= satellite_data.shape[1] or lon_idx < 0 or lon_idx >= satellite_data.shape[
                2]:
            # print(f"Latitude or longitude out of range for station {station_id}")
            continue

        # 将站点的降水值存储到对应网格的字典中

        grid_key = (lat_idx, lon_idx)

        # 如果该网格在grid_data中没有记录，先初始化
        if grid_key not in grid_data:
            grid_data[grid_key] = {}

        # 如果该日期没有记录，初始化为空列表
        if station_date not in grid_data[grid_key]:
            grid_data[grid_key][station_date] = []

        grid_data[grid_key][station_date].append(station_precip)

    # 检查网格数据内容
    # print(f"---网格数据检查---")
    # for grid_key, date_dict in grid_data.items():
    #     print(f"Grid ({grid_key[0]}, {grid_key[1]}) - 日期: {list(date_dict.keys())}")

        # # 如果网格有日期和降水值数据
        # for date, precip_values in date_dict.items():
        #     print(f"日期: {date}, 降水值列表: {precip_values}")

    print(f"---均值计算中--")
    # 计算每个网格每个日期的平均降水值
    for grid_key, date_dict in grid_data.items():
        for date, precip_values in date_dict.items():
            # 计算平均值并替换原有的列表
            avg_precip = np.mean(precip_values)
            grid_data[grid_key][date] = avg_precip

            # 打印每个网格每个日期的平均降水值，检查计算结果
            # print(f"Grid ({grid_key[0]}, {grid_key[1]}) - 日期: {date}, 平均降水值: {avg_precip}")

    # 遍历每个网格，计算该网格所有站点降水值的均值，并与卫星数据对比
    print(f"---匹配计算中---")
    for (lat_idx, lon_idx), date_data in grid_data.items():
        satellite_precip_vals = []  # 用来存储与站点数据匹配的卫星降水值

        # 获取网格的卫星数据
        satellite_precip = satellite_data[:, lat_idx, lon_idx]
        # 打印卫星数据的所有值
        # print(f"未匹配卫星数据 for grid ({lat_idx}, {lon_idx}): {satellite_precip}")

        # # Adjust index for missing dates
        # missing_dates = [
        #     pd.Timestamp("2007-02-25"),
        #     pd.Timestamp("2007-02-26"),
        #     pd.Timestamp("2007-02-27"),
        # ]
        #
        # # 从 `date_data` 中过滤掉缺失的日期
        # filtered_date_data = {date: value for date, value in date_data.items() if date not in missing_dates}
        #
        # # 现在，可以从过滤后的数据中获取降水值列表
        # station_precip_values = list(filtered_date_data.values())

        station_precip_values = list(date_data.values())  # 直接从 `date_data` 中获取均值列表


        # 打印所有日期的站点数据和卫星数据
        # print(f"未匹配站点数据：{station_precip_values}")

        # 遍历该网格下的每个日期
        print(f"---开始日期匹配---")

        for date, precip_list in date_data.items():
            # 打印每个日期的降水值列表
            # print(f"日期: {date}, 站点降水值列表: {precip_list}")

            # 获取日期索引，计算从2007-01-01起的天数
            station_date_idx = (date - pd.Timestamp("2007-01-01")).days

            # if station_date_idx < 0:
            #     print(f"Date out of range for station: on day {station_date_idx}")
            #     continue
            #
            # if date > pd.Timestamp("2007-02-27"):
            #     station_date_idx -= 3
            # print(f"当前处理后的日期索引: {station_date_idx}")
            #
            # if station_date_idx >= satellite_data.shape[0]:
            #     print(f"Date out of range for station: on day {station_date_idx}")
            #     continue

            if station_date_idx < 0 or station_date_idx >= satellite_data.shape[0]:
                print(f"Date out of range for station: on day {station_date_idx}")
                continue

            # 提取卫星数据的降水值
            satellite_precip_val = satellite_precip[station_date_idx]
            satellite_precip_vals.append(satellite_precip_val)

        # print(f"计算前站点数据：{station_precip_values}")
        # print(f"计算前卫星数据{satellite_precip_vals}")

        # 假设你在这里提取了站点数据和卫星数据
        # 将原本的 list 类型转换为 numpy 数组
        station_precip_values = np.array(station_precip_values)
        satellite_precip_vals = np.array(satellite_precip_vals)

        # print(
        #     f"Before nonnan - station_precip_values: {station_precip_values.shape}, satellite_precip_vals: {satellite_precip_vals.shape}")
        #
        # station_precip_values, satellite_precip_vals = nonnan(station_precip_values, satellite_precip_vals)

        # 再次打印它们的形状
        # print(f"Before calculate - station_precip_values: {station_precip_values.shape}, satellite_precip_vals: {satellite_precip_vals.shape}")

        # 计算统计指标（R, RMSE, MAE 等）来比较站点均值和卫星数据
        r_val = R(station_precip_values, satellite_precip_vals)
        rmse_val = RMSE(station_precip_values, satellite_precip_vals)
        mae_val = MAE(station_precip_values, satellite_precip_vals)
        norm_rmse_val = normRMSE(station_precip_values, satellite_precip_vals)
        norm_mae_val = normMAE(station_precip_values, satellite_precip_vals)
        bias_val = Sum(station_precip_values) - Sum(satellite_precip_vals)
        pod_val = POD(station_precip_values, satellite_precip_vals)
        far_val = FAR(station_precip_values, satellite_precip_vals)
        csi_val = CSI(station_precip_values, satellite_precip_vals)

        # 将当前网格的统计结果保存到列表
        stats['lat_idx'].append(lat_idx)
        stats['lon_idx'].append(lon_idx)
        stats['r'].append(r_val)
        stats['rmse'].append(rmse_val)
        stats['mae'].append(mae_val)
        stats['norm_rmse'].append(norm_rmse_val)
        stats['norm_mae'].append(norm_mae_val)
        stats['bias'].append(bias_val)
        stats['pod'].append(pod_val)
        stats['far'].append(far_val)
        stats['csi'].append(csi_val)

        # 打印当前网格的统计结果
        print(f"Grid (Lat: {lat_idx}, Lon: {lon_idx}) ")
        print(f"  R: {r_val}")
        print(f"  RMSE: {rmse_val}")
        print(f"  MAE: {mae_val}")
        print(f"  Norm RMSE: {norm_rmse_val}")
        print(f"  Norm MAE: {norm_mae_val}")
        print(f"  Bias: {bias_val}")
        print(f"  POD: {pod_val}")
        print(f"  FAR: {far_val}")
        print(f"  CSI: {csi_val}")
        print("-" * 50)

    # 将所有统计结果保存到 CSV 文件
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(output_file, index=False, mode='a', header=not os.path.exists(output_file))  # 追加写入
    print(f"---计算完成，结果已保存到 {output_file}---")


if __name__ == '__main__':
    start_time = time.time()  # 记录开始时间

    station_folder = 'D:/hongyouting/data/station/GSOD4_nan'
    satellite_file = 'D:/hongyouting/data/0.25_clip_60/gsmap_nrtg_precipitation.npz'
    output_file = 'D:/hongyouting/result/station_statistics/1/station_gsmap_nrtg.csv'

    # station_folder = 'D:/hongyouting/data/station/GSOD4_nan'
    # satellite_file = 'D:/hongyouting/data/0.25_clip_60/gsmap_nrtg_precipitation.npz'
    # output_file = 'D:/hongyouting/result/station_statistics/2/station_gsmap_nrtg.csv'

    # station_folder = 'D:/hongyouting/data/station/GSOD4_nan'
    # satellite_file = 'D:/hongyouting/data/0.25_clip_60/cmorph_precipitation.npz'
    # output_file = 'D:/hongyouting/result/station_statistics/1/station_cmorph.csv'

    # station_folder = 'D:/hongyouting/data/station/GSOD4_nan'
    # satellite_file = 'D:/hongyouting/data/0.25_clip_60/cdr_precipitation.npz'
    # output_file = 'D:/hongyouting/result/station_statistics/1/station_cdr.csv'

    # station_folder = 'D:/hongyouting/data/station/GSOD4_nan'
    # satellite_file = 'D:/hongyouting/data/0.25_clip_50/4930/chirps_precipitation.npz'
    # output_file = 'D:/hongyouting/result/station_statistics/1/station_chirps.csv'

    # station_folder = 'D:/hongyouting/data/station/GSOD4_nan'
    # satellite_file = 'D:/hongyouting/data/0.25_clip/imerg_f7_4930_precipitation.npz'
    # output_file = 'D:/hongyouting/result/station_statistics/1/station_imerg_f7.csv'

    # station_folder = 'D:/hongyouting/data/station/GSOD4_nan'
    # satellite_file = 'D:/hongyouting/data/0.25_clip/imerg_e7_4930_precipitation.npz'
    # output_file = 'D:/hongyouting/result/station_statistics/1/station_imerg_e7.csv'

    # station_folder = 'D:/hongyouting/data/station/GSOD4_nan'
    # satellite_file = 'D:/hongyouting/data/0.25_clip_50/4930/sm2rain_precipitation.npz'
    # output_file = 'D:/hongyouting/result/station_statistics/1/station_sm2rain.csv'

    # station_folder = 'D:/hongyouting/data/station/GSOD4_nan'
    # satellite_file = 'D:/hongyouting/data/0.25_clip_50/4930/era5_precipitation.npz'
    # output_file = 'D:/hongyouting/result/station_statistics/1/station_era5.csv'

    station_data = read_station_data(station_folder)
    satellite_data = read_satellite_data(satellite_file)

    calculate_statistics(station_data, satellite_data, output_file=output_file)

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算程序运行时间

    print(f"程序运行时间: {elapsed_time:.2f} 秒")

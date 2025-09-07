import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import time


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

def read_station_data(folder_path):
    station_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.getsize(file_path) > 0:  # Check if the file is not empty
                    data = pd.read_csv(file_path, header=None)
                else:
                    print(f"Skipped empty file: {filename}")
                    continue

                # Filter dates greater than 2020/6/30
                data[1] = pd.to_datetime(data[1], format='%Y/%m/%d', errors='coerce')
                data = data[data[1] <= pd.Timestamp('2020-06-30')]

                if len(data) >= 365:  # Check if data has at least 365 rows
                    station_data.append(data)
            except pd.errors.EmptyDataError:
                print(f"Skipped empty file: {filename}")
                continue

    if station_data:
        station_data = pd.concat(station_data, ignore_index=True)
    else:
        station_data = pd.DataFrame()  # Return an empty DataFrame if no valid data files are found
    return station_data


def read_satellite_data(file_path):
    data = np.load(file_path)
    return data


def calculate_statistics(station_data, satellite_data, lat_range, lon_range, output_files, min_samples=10):
    levels = {
        'light_rain': (0, 10, output_files[0]),
        'moderate_rain': (10, 25, output_files[1]),
        'heavy_rain': (25, float('inf'), output_files[2])
    }

    station_data[4] = station_data[4].replace(2539.746, np.nan)
    station_ids = station_data[0].unique()

    for station_id in station_ids:
        stats_template = {  # Initialize stats_template for each station
            'station_id': [],
            'rmse': [],
            'r': [],
            'mae': [],
            'norm_rmse': [],
            'norm_mae': [],
            'bias': [],
            'pod': [],
            'far': [],
            'csi': []
        }

        station_subset = station_data[station_data[0] == station_id]

        for level, (min_val, max_val, output_file) in levels.items():
            station_precips = []
            satellite_precips = []

            for _, row in station_subset.iterrows():
                station_date = pd.to_datetime(row[1], format='%Y/%m/%d', errors='coerce')
                if pd.isna(station_date):
                    continue

                station_lat = row[2]
                station_lon = row[3]
                station_precip = row[4]

                if np.isnan(station_precip):
                    continue

                if station_lat < -60 or station_lat > 60:
                    continue

                day_of_year = (station_date - pd.Timestamp("2007-01-01")).days

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
                    # print(f"Date out of range for station {station_id} on day {day_of_year}")
                    continue

                satellite_precip = satellite_data[day_of_year, :, :]

                lat_idx = np.abs(lat_range - station_lat).argmin()
                lon_idx = np.abs(lon_range - station_lon).argmin()

                if lat_idx < 0 or lat_idx >= satellite_precip.shape[0] or lon_idx < 0 or lon_idx >= \
                        satellite_precip.shape[1]:
                    continue

                satellite_precip_value = satellite_precip[lat_idx, lon_idx]

                if np.isnan(satellite_precip_value):
                    continue

                # Determine rain level and add data to corresponding level
                if min_val <= station_precip < max_val:
                    station_precips.append(station_precip)
                    satellite_precips.append(satellite_precip_value)

            if len(station_precips) < min_samples:
                continue

            station_precips_array = np.array(station_precips)
            satellite_precips_array = np.array(satellite_precips)

            rmse_val = RMSE(station_precips_array, satellite_precips_array)
            mae_val = MAE(station_precips_array, satellite_precips_array)
            norm_rmse_val = normRMSE(station_precips_array, satellite_precips_array)
            norm_mae_val = normMAE(station_precips_array, satellite_precips_array)
            r_val = R(station_precips_array, satellite_precips_array)
            bias_val = totalVolumeRatio(station_precips_array, satellite_precips_array)
            pod_val = POD(station_precips_array, satellite_precips_array)
            far_val = FAR(station_precips_array, satellite_precips_array)
            csi_val = CSI(station_precips_array, satellite_precips_array)

            stats_template['station_id'].append(station_id)
            stats_template['rmse'].append(rmse_val)
            stats_template['r'].append(r_val)
            stats_template['mae'].append(mae_val)
            stats_template['norm_rmse'].append(norm_rmse_val)
            stats_template['norm_mae'].append(norm_mae_val)
            stats_template['bias'].append(bias_val)
            stats_template['pod'].append(pod_val)
            stats_template['far'].append(far_val)
            stats_template['csi'].append(csi_val)

            print(f"Station ID: {station_id}, Level: {level}, "
                  f"RMSE: {rmse_val}, R: {r_val}, MAE: {mae_val}, "
                  f"normRMSE: {norm_rmse_val}, normMAE: {norm_mae_val}, "
                  f"BIAS: {bias_val}, POD: {pod_val}, FAR: {far_val}, "
                  f"CSI: {csi_val}")

            df = pd.DataFrame(stats_template)
            df.to_csv(output_file, index=False, mode='a', header=not os.path.exists(output_file))

            # Clear stats_template after saving to CSV
            stats_template = {
                'station_id': [],
                'rmse': [],
                'r': [],
                'mae': [],
                'norm_rmse': [],
                'norm_mae': [],
                'bias': [],
                'pod': [],
                'far': [],
                'csi': []
            }

    return stats_template


if __name__ == '__main__':
    start_time = time.time()

    # station_folder = 'D:/hongyouting/data/GSOD1'
    # satellite_file = 'D:/hongyouting/data/0.25_clip_60/gsmap_precipitation.npy'
    # output_files = [
    #     'D:/hongyouting/result/station_statistics/categorization50/gsmap_light.csv',
    #     'D:/hongyouting/result/station_statistics/categorization50/gsmap_moderate.csv',
    #     'D:/hongyouting/result/station_statistics/categorization50/gsmap_heavy.csv'
    # ]

    station_folder = 'D:/hongyouting/data/GSOD'
    satellite_file = 'D:/hongyouting/data/0.25_clip_60/gsmap_nrt_precipitation.npy'
    output_files = [
            'D:/hongyouting/result/station_statistics/categorization50/gsmap_nrt_light.csv',
            'D:/hongyouting/result/station_statistics/categorization50/gsmap_nrt_moderate.csv',
            'D:/hongyouting/result/station_statistics/categorization50/gsmap_nrt_heavy.csv'
    ]

    station_data = read_station_data(station_folder)
    satellite_data = read_satellite_data(satellite_file)

    lat_range = np.linspace(60, -60, 480)
    lon_range = np.linspace(-180, 180, 1440)

    calculate_statistics(station_data, satellite_data, lat_range, lon_range, output_files, min_samples=50)

    end_time = time.time()
    print(f'程序运行时间: {end_time - start_time} 秒')
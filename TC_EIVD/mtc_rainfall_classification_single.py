import numpy as np
import pandas as pd
from tqdm import tqdm  # 导入tqdm库
import time

# 记录整个程序开始时间
start_time = time.time()

# 记录加载数据开始时间
load_start_time = time.time()
# 加载.npy文件
# mswep_file = 'D:/hongyouting/data/0.25_clip/test/mswep/mswep_precipitation.npy'
# era5_file = 'D:/hongyouting/data/0.25_clip/test/era5/era5_precipitation.npy'
# sm2rain_file = 'D:/hongyouting/data/0.25_clip/test/sm2rain/sm2rain_precipitation.npy'
# satellite_file = 'D:/hongyouting/data/0.25_clip/test/imerg/imerg_precipitation.npy'
mswep_file = 'D:/hongyouting/data/0.25_clip/mswep_precipitation.npy'
era5_file = 'D:/hongyouting/data/0.25_clip/era5_precipitation.npy'
sm2rain_file = 'D:/hongyouting/data/0.25_clip/sm2rain_precipitation.npy'
satellite_file = 'D:/hongyouting/data/0.25_clip/imerg_precipitation.npy'

# 加载数据
print("Loading data...")
mswep_data = np.load(mswep_file)
era5_data = np.load(era5_file)
sm2rain_data = np.load(sm2rain_file)
satellite_data = np.load(satellite_file)

# 计算加载数据所花费的时间
load_end_time = time.time()
load_elapsed_time = load_end_time - load_start_time
print(f"Time for loading data: {load_elapsed_time:.2f} seconds")
print("Load successfully ")

class TCAnalysis:
    def __init__(self, ts_era5, ts_sm2rain, ts_satellite):
        # 初始化函数，接受 ERA5、SM2RAIN 和 MSWEP 时间序列数据
        self.ts_era5 = ts_era5
        self.ts_sm2rain = ts_sm2rain
        self.ts_satellite = ts_satellite

    def single(self, row_idx, col_idx, ts_data):
        # 从元组中提取时间序列数据
        ts_era5, ts_sm2rain, ts_satellite = ts_data

        print('processing (%d,%d)' % (row_idx, col_idx))  # 打印正在处理的网格点的行列索引

        # Create arrays to store RMSE and CC
        RMSE = np.zeros((1, 3), dtype=np.float16)
        CC = np.zeros((1, 3), dtype=np.float16)

        # # 从矩阵中读取时间序列数据
        # ts_era5 = self.ts_era5[:, row_idx, col_idx]
        # ts_sm2rain = self.ts_sm2rain[:, row_idx, col_idx]
        # ts_satellite = self.ts_satellite[:, row_idx, col_idx]

        ts = pd.DataFrame({
            'era5': ts_era5,
            'sm2rain': ts_sm2rain,
            'satellite': ts_satellite
        })

        # # 做时间序列数据的处理
        # ts.era5 = ts.era5.shift()
        # ts.satellite = ts.satellite.shift()
        # ts.sm2rain = ts.sm2rain.shift()

        ts = self.preprocess(ts)
        ts.replace(-9999, np.nan, inplace=True)  # 将 -9999 替换为 NaN
        # ts[ts < 0] = 0

        if len(ts) == 0 or len(ts) < 3:
            RMSE[0, :] = np.nan
            CC[0, :] = np.nan
        else:
            _sig, _r = self.mtc(ts)
            RMSE[0, :] = _sig
            CC[0, :] = _r

        # 返回RMSE和CC的值
        # print(f"Grid Point ({row_idx}, {col_idx}):")
        print(f"RMSE: {RMSE}")
        print(f"CC: {CC}")
        print("--------------------")

        return RMSE, CC

    def preprocess(self, data, threshold1=0, threshold2=0.001):
        # this function drops nan and values between thresholds
        data= data.astype('float32')
        cols= data.columns
        for col in cols:
            data=data[data[col]>=threshold1]
            data.clip(lower=threshold2, inplace=True)
            # data.dropna(inplace=True)

        data= data.apply(np.log)
        return data

        # print("Processed Data:")
        # print(data)


        # mtc计算
    def mtc(self, X):
        # # 检查 X 的形状是否为 (时间序列, 3)
        # if len(X) < 3:
        #     # 如果长度小于3，直接返回无效值
        #     return np.array([-9999] * 3), np.array([-9999] * 3)

        N_boot = 500  # 设置 Bootstrap 重采样的次数为 500。

        # 创建一个数组，用于存储每次 Bootstrap 重采样中计算得到的 rmse、cc
        rmse = np.zeros((N_boot, 3))
        cc = np.zeros((N_boot, 3))

        # 遍历 Bootstrap 重采样次数
        for i in range(N_boot):
            # 创建一个数组，用于存储每次 Bootstrap 重采样中计算得到的标准差和相关系数
            sigma = np.zeros(3)
            r = np.zeros(3)

            sample = self.bootstrap_resample(X, n=N_boot)  # 调用 bootstrap_resample 方法进行 Bootstrap 重采样，获取一个样本。
            # print('样本数据:')
            # print(sample)

            cov = sample.cov().to_numpy()  # 计算样本的协方差矩阵
            # print('协方差矩阵:')
            # print(cov)

            # 计算 RMSE
            # 检查协方差矩阵是否包含零值
            if (cov == 0).any().any():  # 存在零值，将对应位置的 RMSE 和 CC 设置为 NaN
                rmse[i, :] = np.nan
                cc[i, :] = np.nan
            else:
                # 计算每个数据集的标准差
                sigma[0] = cov[0, 0] - (cov[0, 1] * cov[0, 2]) / (cov[1, 2])
                sigma[1] = cov[1, 1] - (cov[0, 1] * cov[1, 2]) / (cov[0, 2])
                sigma[2] = cov[2, 2] - (cov[0, 2] * cov[1, 2]) / (cov[0, 1])

                # 将标准差中小于 0 的值设为 NaN，并取平方根
                sigma[sigma < 0] = np.nan
                # sigma[sigma == 0] = 1e-3  # 添加一个小值，避免除以零
                sigma = sigma ** 0.5

                # 计算每个数据集之间的相关系数
                r[0] = (cov[0, 1] * cov[0, 2]) / (cov[0, 0] * cov[1, 2])
                r[1] = (cov[0, 1] * cov[1, 2]) / (cov[1, 1] * cov[0, 2])
                r[2] = (cov[0, 2] * cov[1, 2]) / (cov[2, 2] * cov[0, 1])

                # 将小于 0 的相关系数设为 0.0001，将大于 1 的相关系数设为 1，并取平方根
                r[r < 0] = 0.0001
                r[r > 1] = 1
                r = r ** 0.5
                r[r < 1e-3] = 0

                # 将计算得到的标准差和相关系数存储到对应位置
                rmse[i, :] = sigma
                cc[i, :] = r
                # print('标准差:')
                # print(sigma)
                # print('相关系数:')
                # print(r)

        # if np.isnan(np.sum(rmse)) or np.isnan(np.sum(cc)):
        #     return np.array([-9999] * 3), np.array([-9999] * 3)
        # else:
        return np.nanmean(rmse, axis=0), np.nanmean(cc, axis=0)

    #执行 Bootstrap 重采样的函数
    def bootstrap_resample(self, X, n=None):

        if n == None:
            n = len(X)

        resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
        X_resample = X.iloc[resample_i, :]

        # print('重采样:')
        # print(sample)

        return X_resample

# 创建 TCAnalysis 实例
tc_analyze = TCAnalysis(era5_data, sm2rain_data, satellite_data)

# 创建空数组来存储所有网格点的RMSE和CC的均值
grid_shape = mswep_data.shape[1:]
results_shape = (np.prod(grid_shape), 6)
# results = np.zeros(results_shape)
results = np.full(results_shape, np.nan)  # 使用NaN填充数组

# 循环遍历所有行和列
index = 0
for row_idx in tqdm(range(grid_shape[0])):
    for col_idx in range(grid_shape[1]):
        grid_values = mswep_data[:, row_idx, col_idx]
        valid_dates = []
        for date_idx, value in enumerate(grid_values):
            # if 0 < value < 10:
            if 10 <= value < 25:
            # if 25 <= value < 50:

                valid_dates.append(date_idx)
        if valid_dates:
            ts_era5 = era5_data[valid_dates, row_idx, col_idx]
            ts_sm2rain = sm2rain_data[valid_dates, row_idx, col_idx]
            ts_satellite = satellite_data[valid_dates, row_idx, col_idx]
            RMSE, CC = tc_analyze.single(row_idx, col_idx, (ts_era5, ts_sm2rain, ts_satellite))
            # if not np.isnan(RMSE).any() and not np.isnan(CC).any():
            #     mean_results[index] = np.concatenate([RMSE[0], CC[0]])
            # else:
            #     mean_results[index] = np.nan
            results[index] = np.concatenate([RMSE[0], CC[0]])
            index += 1

# 将结果保存到CSV文件中
df = pd.DataFrame(results, columns=['RMSE_era5', 'RMSE_sm2rain', 'RMSE_satellite', 'CC_era5', 'CC_sm2rain', 'CC_satellite'])
# csv_filename = 'D:/result/TCresults/categorization/esi_light_results.csv'
csv_filename = 'D:/result/TCresults/categorization/esi_moderate_results.csv'
# csv_filename = 'D:/result/TCresults/categorization/esi_heavy_results.csv'

df.to_csv(csv_filename, index=False, na_rep='NaN')  # 将NaN替换为字符串'NaN'
print(f"Results saved to {csv_filename}")

# 打印结果数组的形状和内容
print("Shape of results array before removing NaN rows:", results.shape)
print("Results array before removing NaN rows:")
print(results)

# 将结果写入文本文件
# with open('D:/result/TCresults/categorization/light_result.txt', 'w') as file:
with open('D:/result/TCresults/categorization/moderate_result.txt', 'w') as file:
# with open('D:/result/TCresults/categorization/heavy_result.txt', 'w') as file:

    file.write("RMSE_era5, RMSE_sm2rain, RMSE_satellite, CC_era5, CC_sm2rain, CC_satellite\n")
    for row in results:
        line = ', '.join(map(str, row)) + '\n'
        file.write(line)

# print("Results before calculating mean written to light_result.txt")
print("Results before calculating mean written to moderate_result.txt")
# print("Results before calculating mean written to heavy_result.txt")


# 结果
# 将小于等于0的值替换为NaN
results[results <= 0] = np.nan
results = results[~np.isnan(results).any(axis=1)]

# 打印结果数组的形状和内容
print("Shape of results array after removing NaN rows:", results.shape)
print("Results array after removing NaN rows:")
print(results)

# 均值
mean_RMSE = np.nanmean(results[:, :3], axis=0)
mean_CC = np.nanmean(results[:, 3:], axis=0)

print("Mean RMSE:", mean_RMSE)
print("Mean CC:", mean_CC)

# 记录整个程序结束时间
end_time = time.time()

# 计算整个程序运行时间并打印
total_elapsed_time = end_time - start_time
print(f"Total time for program: {total_elapsed_time:.2f} seconds")
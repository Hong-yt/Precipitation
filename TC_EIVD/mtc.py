'''
Triple Collocation Method
'''

#tc计算—已修改--main运行

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from osgeo import gdal
import multiprocessing
from functools import partial
os.environ['PROJ_LIB'] = r'G:\data\venv\Lib\site-packages\osgeo\data\proj'

class TripleCollocation(object):

    def __init__(self):

        # 初始化属性，避免 AttributeError
        self.ts_era5 = None
        self.ts_sm2rain = None
        self.ts_satellite = None

    def read_npy_files(self):
        # 一次性读取所有 .npy 文件
        self.ts_era5 = np.load('D:/hongyouting/data/0.25_clip/era5_precipitation.npy')
        self.ts_sm2rain = np.load('D:/hongyouting/data/0.25_clip/sm2rain_precipitation.npy')
        self.ts_satellite = np.load('D:/hongyouting/data/0.25_clip/imerg_f7_4926_precipitation.npy')
        # self.ts_satellite = np.load('D:/hongyouting/data/0.25_clip/imerg_e7_4926_precipitation.npy')

        # self.ts_era5 = np.load('D:/hongyouting/data/0.25_clip_60/era5_precipitation.npy')
        # self.ts_sm2rain = np.load('D:/hongyouting/data/0.25_clip_60/sm2rain_precipitation.npy')
        # self.ts_satellite = np.load('D:/hongyouting/data/0.25_clip_60/cdr_precipitation.npy')

        # self.ts_era5 = np.load('D:/hongyouting/data/0.25_clip_50/era5_precipitation.npy')
        # self.ts_sm2rain = np.load('D:/hongyouting/data/0.25_clip_50/sm2rain_precipitation.npy')
        # self.ts_satellite = np.load('D:/hongyouting/data/0.25_clip_50/chirps_precipitation.npy')

        # self.ts_era5 = np.load('D:/hongyouting/data/0.25_clip_60/era5_precipitation.npy')
        # self.ts_sm2rain = np.load('D:/hongyouting/data/0.25_clip_60/sm2rain_precipitation.npy')
        # self.ts_satellite = np.load('D:/hongyouting/data/0.25_clip_60/cmorph_precipitation.npy')
        # self.ts_satellite = np.load('D:/hongyouting/data/0.25_clip_60/gsmap_nrt_precipitation.npy')
        # self.ts_satellite = np.load('D:/hongyouting/data/0.25_clip_60/gsmap_nrtg_precipitation.npy')

        # self.ts_era5 = np.load('D:/hongyouting/data/0.25_clip_60/1/era5_precipitation.npy')
        # self.ts_sm2rain = np.load('D:/hongyouting/data/0.25_clip_60/1/sm2rain_precipitation.npy')
        # self.ts_satellite = np.load('D:/hongyouting/data/0.25_clip_60/1/gsmap_mvkg_precipitation.npy')

    # 处理每个网格点的时间序列数据，计算均方根误差（RMSE）和相关系数（CC）
    def main(self):
        # i = 400
        # i = 480
        i = 720
        j = 1440

        # Create lists to store RMSE and CC along with grid coordinates
        data_list = []

        for ii in range(i):
            for jj in range(j):
                print('processing (%d,%d)' % (ii, jj))

                ts_era5 = self.ts_era5[:, ii, jj]
                ts_sm2rain = self.ts_sm2rain[:, ii, jj]
                ts_satellite = self.ts_satellite[:, ii, jj]

                ts = pd.DataFrame({
                    'era5': ts_era5,
                    'sm2rain': ts_sm2rain,
                    'satellite': ts_satellite
                })

                ts = self.preprocess(ts)
                # ts.replace(-9999, np.nan, inplace=True)  # 将 -9999 替换为 NaN
                # ts[ts < 0] = 0

                if len(ts) == 0 or len(ts) < 3:
                    rmse_values = [np.nan, np.nan, np.nan]
                    cc_values = [np.nan, np.nan, np.nan]
                else:
                    _sig, _r = self.mtc(ts)
                    rmse_values = _sig
                    cc_values = _r

                    print(f"Grid Point ({ii}, {jj}):")
                    print(f"RMSE: {_sig}")
                    print(f"CC: {_r}")
                    print("--------------------")

                # Append RMSE and CC values along with the grid coordinates
                data_list.append([ii, jj] + list(rmse_values) + list(cc_values))

        # Create DataFrame for RMSE and CC including grid coordinates
        columns = ['i', 'j', 'RMSE_era5', 'RMSE_sm2rain', 'RMSE_satellite', 'CC_era5', 'CC_sm2rain', 'CC_satellite']
        df = pd.DataFrame(data_list, columns=columns)

        # Save the DataFrame to a CSV file
        df.to_csv('D:/hongyouting/result/TCresults/1/esimerg_f7_results.csv', index=False)
        # df.to_csv('D:/hongyouting/result/TCresults/1/esimerg_e7_results.csv', index=False)
        # df.to_csv('D:/hongyouting/result/TCresults/1/escdr_results.csv', index=False)
        # df.to_csv('D:/hongyouting/result/TCresults/1/eschirps_results.csv', index=False)
        # df.to_csv('D:/hongyouting/result/TCresults/1/esgsmap_mvkg_results.csv', index=False)
        # df.to_csv('D:/hongyouting/result/TCresults/1/escmorph_results.csv', index=False)
        # df.to_csv('D:/hongyouting/result/TCresults/1/esgsmap_nrt_results.csv', index=False)

        # Compute and print mean values
        mean_RMSE_era5 = np.nanmean(df['RMSE_era5'])
        mean_RMSE_sm2rain = np.nanmean(df['RMSE_sm2rain'])
        mean_RMSE_sat = np.nanmean(df['RMSE_satellite'])

        mean_CC_era5 = np.nanmean(df['CC_era5'])
        mean_CC_sm2rain = np.nanmean(df['CC_sm2rain'])
        mean_CC_sat = np.nanmean(df['CC_satellite'])

        print(f"Mean (RMSE for era5, sm2rain, satellite): ({mean_RMSE_era5}, {mean_RMSE_sm2rain}, {mean_RMSE_sat})")
        print(f"Mean (CC for era5, sm2rain, satellite): ({mean_CC_era5}, {mean_CC_sm2rain}, {mean_CC_sat})")

        # # 输出 TIF 文件
        # self.write_geotif('G:/py_data/TCresults/rmse_era5.tif', RMSE_era5)
        # self.write_geotif('G:/py_data/TCresults/rmse_sat.tif', RMSE_sat)
        # self.write_geotif('G:/py_data/TCresults/rmse_sm2rain.tif', RMSE_sm2rain)
        #
        # self.write_geotif('G:/py_data/TCresults/cc_era5.tif', CC_era5)
        # self.write_geotif('G:/py_data/TCresults/cc_sat.tif', CC_sat)
        # self.write_geotif('G:/py_data/TCresults/cc_sm2rain.tif', CC_sm2rain)

    def main_single(self):

        # 选择要处理的网格点
        target_i = 360
        target_j = 480

        # Create arrays to store RMSE and CC
        RMSE = np.zeros((1, 3), dtype=np.float16)
        CC = np.zeros((1, 3), dtype=np.float16)

        ts_era5 = self.ts_era5[:, target_i, target_j]
        ts_satellite = self.ts_satellite[:, target_i, target_j]
        ts_sm2rain = self.ts_sm2rain[:, target_i, target_j]

        ts = pd.DataFrame({
            'era5': ts_era5,
            'satellite': ts_satellite,
            'sm2rain': ts_sm2rain
        })

        ts = self.preprocess(ts)
        ts.replace(-9999, np.nan, inplace=True)  # 将 -9999 替换为 NaN
        ts[ts < 0] = 0

        if len(ts) == 0 or len(ts) < 3:
            RMSE[0, :] = np.nan
            CC[0, :] = np.nan
        else:
            _sig, _r = self.mtc(ts)
            RMSE[0, :] = _sig
            CC[0, :] = _r

            print(f"Grid Point ({target_i}, {target_j}):")
            print(f"RMSE: {_sig}")
            print(f"CC: {_r}")
            print("--------------------")

        # 输出均值结果
        mean_RMSE_era5 = np.nanmean(RMSE[:, 0])
        mean_RMSE_sat = np.nanmean(RMSE[:, 1])
        mean_RMSE_sm2rain = np.nanmean(RMSE[:, 2])

        mean_CC_era5 = np.nanmean(CC[:, 0])
        mean_CC_sat = np.nanmean(CC[:, 1])
        mean_CC_sm2rain = np.nanmean(CC[:, 2])

        # Create DataFrame for detailed values
        detailed_values = pd.DataFrame({
            'Grid_RMSE_era5': [RMSE[0, 0]],
            'Grid_RMSE_satellite': [RMSE[0, 1]],
            'Grid_RMSE_sm2rain': [RMSE[0, 2]],
            'Grid_CC_era5': [CC[0, 0]],
            'Grid_CC_satellite': [CC[0, 1]],
            'Grid_CC_sm2rain': [CC[0, 2]]
        })

        # Create DataFrame for mean values
        mean_values = pd.DataFrame({
            'Mean_RMSE_era5': [np.nanmean(RMSE[:, 0])],
            'Mean_RMSE_satellite': [np.nanmean(RMSE[:, 1])],
            'Mean_RMSE_sm2rain': [np.nanmean(RMSE[:, 2])],
            'Mean_CC_era5': [np.nanmean(CC[:, 0])],
            'Mean_CC_satellite': [np.nanmean(CC[:, 1])],
            'Mean_CC_sm2rain': [np.nanmean(CC[:, 2])]
        })

        # Concatenate DataFrames horizontally
        combined_values = pd.concat([detailed_values, mean_values], axis=1)

        # Save combined DataFrame to CSV file
        combined_values.to_csv('D:/hongyouting/result/TCresults/test_results.csv', index=False)

        # 输出均值结果
        print("Mean RMSE for era5:", mean_RMSE_era5)
        print("Mean RMSE for satellite:", mean_RMSE_sat)
        print("Mean RMSE for sm2rain:", mean_RMSE_sm2rain)

        print("Mean CC for era5:", mean_CC_era5)
        print("Mean CC for satellite:", mean_CC_sat)
        print("Mean CC for sm2rain:", mean_CC_sm2rain)

    #使用并行计算的方式处理每个网格点的数据
    def parallel(self, cores=6, write=True):
        RMSE = np.zeros((720, 1440, 3), dtype=np.float16)
        CC = np.zeros((720, 1440, 3), dtype=np.float16)

        inputs = [(i, j) for i in range(720) for j in range(1440)]
        pool = multiprocessing.Pool(cores)
        results = pool.map(self.single, inputs)

        for r, c, i, j in results:
            RMSE[i, j, :] = r
            CC[i, j, :] = c

        RMSE_era5 = RMSE[:, :, 0]
        RMSE_sat = RMSE[:, :, 1]
        RMSE_sm2rain = RMSE[:, :, 2]

        CC_era5 = CC[:, :, 0]
        CC_sat = CC[:, :, 1]
        CC_sm2rain = CC[:, :, 2]

        print('Mean RMSE for era5:', np.nanmean(RMSE_era5))
        print('Mean RMSE for satellite:', np.nanmean(RMSE_sat))
        print('Mean RMSE for sm2rain:', np.nanmean(RMSE_sm2rain))

        print('Mean CC for era5:', np.nanmean(CC_era5))
        print('Mean CC for satellite:', np.nanmean(CC_sat))
        print('Mean CC for sm2rain:', np.nanmean(CC_sm2rain))

        # if write:
        #     self.write_geotif('G:/py_data/TCresults_pa/rmse_era5.tif',RMSE_era5)
        #     self.write_geotif('G:/py_data/TCresults_pa/rmse_sat.tif',  RMSE_sat)
        #     self.write_geotif('G:/py_data/TCresults_pa/rmse_sm2rain.tif',RMSE_sm2rain)
        #
        #     self.write_geotif('G:/py_data/TCresults_pa/cc_era5.tif',CC_era5)
        #     self.write_geotif('G:/py_data/TCresults_pa/cc_sat.tif',  CC_sat)
        #     self.write_geotif('G:/py_data/TCresults_pa/cc_sm2rain.tif',CC_sm2rain)

        return RMSE, CC

    #处理单个网格点的数据,执行 Triple Collocation (TC) 计算
    def single(self, args):
        i, j = args  # 从元组中提取行列索引
        print('processing (%d,%d)' % (i, j))  # 正在处理的网格点的行列索引

        # 从矩阵中读取时间序列数据
        ts_era5 = self.ts_era5[:, i, j]
        ts_satellite = self.ts_satellite[:, i, j]
        ts_sm2rain = self.ts_sm2rain[:, i, j]

        ts = pd.DataFrame({
            'era5': ts_era5,
            'satellite': ts_satellite,
            'sm2rain': ts_sm2rain
        })

        # 做时间序列数据的处理
        ts.era5 = ts.era5.shift()
        ts.satellite = ts.satellite.shift()
        ts.sm2rain = ts.sm2rain.shift()

        ts = self.preprocess(ts)
        # ts[ts < 0] = 0

        # print('length of ts: ', len(ts))
        if len(ts) == 0 or len(ts) < 3:
            RMSE = np.array([-9999] * 3)
            CC = np.array([-9999] * 3)
        else:
            # print(ts.columns)
            RMSE, CC = self.mtc(ts)

        return RMSE, CC, i, j

    #预处理输入的时间序列数据
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

    #mtc计算
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

    # #将 NumPy 数组写入 GeoTIFF 文件的函数
    # def write_geotif(self, dst, new_array):#类的实例、目标路径、numpy数组
    #     #read sample
    #     pth= 'G:/mswep20070101.tif'#样本文件路径
    #     sample= gdal.Open(pth)
    #     projection= sample.GetProjection()
    #     trans= sample.GetGeoTransform()
    #     bands= 1
    #     driver= gdal.GetDriverByName("GTiff")
    #     rows, cols= new_array.shape
    #
    #     outdata = driver.Create(dst, cols, rows, bands, gdal.GDT_Float32)
    #
    #     outdata.SetGeoTransform(trans)
    #     outdata.SetProjection(projection)
    #     outdata.GetRasterBand(bands).WriteArray(new_array)
    #     outdata.FlushCache()
    #     outdata = None

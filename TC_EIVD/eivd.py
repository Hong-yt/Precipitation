# 分类
import numpy as np
import pandas as pd
import os
import time


def EIVD(tri):
    # Check the size of the input matrix
    if tri.shape[0] < 4:
        weight = np.full((1, 3), np.nan)
        sca_tri = np.full((1, 3), np.nan)
        return weight, sca_tri

    # Perform rescaling on input data, 重缩放、计算协方差矩阵
    ExxT_unres = np.cov(tri, rowvar=False)
    sca_tri = np.zeros_like(tri)  # 创建一个与输入矩阵大小相同的零矩阵
    sca_tri[:, 0] = tri[:, 0]  # 第一列保持不变
    # 对第二列和第三列进行重缩放
    sca_tri[:, 1] = (ExxT_unres[0, 2] / ExxT_unres[1, 2]) * (tri[:, 1] - np.nanmean(tri[:, 1])) + np.nanmean(tri[:, 0])
    sca_tri[:, 2] = (ExxT_unres[0, 1] / ExxT_unres[2, 1]) * (tri[:, 2] - np.nanmean(tri[:, 2])) + np.nanmean(tri[:, 0])

#协方差矩阵
    ExxT = np.cov(sca_tri, rowvar=False)

    # Generate Lag-1 [tem-res] series 生成滞后1的[时间-残差]序列
    L_tri = sca_tri[:-1]
    tri_lag = sca_tri[1:]
    L = np.nanmean(L_tri * tri_lag, axis=0)

    # Calculate EIVD results 构造用于计算EIVD结果的线性方程组
    A = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                  [0, 1, 0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0, 1, 0],
                  [0, 0, 0, 1, 0, 0, 0, 1],
                  [1, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0]])

    # 计算线性方程组的值
    y = np.hstack([np.diag(ExxT), ExxT[1, 2], ExxT[0, 1] * np.sqrt(L[0] / L[1]),
                   ExxT[0, 2] * np.sqrt(L[0] / L[2]), ExxT[1, 0] * np.sqrt(L[1] / L[0]),
                   ExxT[2, 0] * np.sqrt(L[2] / L[0]), ExxT[0, 1] * np.sqrt(L[2] / L[0]),
                   ExxT[0, 2] * np.sqrt(L[1] / L[0])])

    x = np.linalg.inv(A.T @ A) @ A.T @ y# 通过最小二乘法求解线性方程组的解

    # Assemble error variance matrix
    # 组装误差方差矩阵，计算标准误差、相关系数、信噪比、fMSE
    EeeT = np.diag(x[4:7])
    EeeT[1, 2] = x[-1]
    EeeT[2, 1] = x[-1]

    stderr = np.sqrt(x[4:7])

    SNR = x[:3] / x[4:7]
    valid_indices = SNR > 0
    SNR = SNR[valid_indices]

    rho2 = SNR / (1 + SNR)
    fMSE = 1- rho2
    rho = np.sqrt(rho2)

    #
    # # 修正相关系数，将其值限制在 [-1, 1] 之间
    # rho = np.clip(rho, -1, 1)

    ecc_values = x[7] / np.sqrt(x[5] * x[6])

    # Solve weight using matrix
    # eta = np.ones((EeeT.shape[1], 1))
    # weight = np.linalg.solve(EeeT, eta) / (eta.T @ (np.linalg.solve(EeeT, eta)))

    # Calculate merged result
    # merged_result = np.sum(sca_tri * weight.T, axis=1)

    # return EeeT, SNR, rho2, fMSE, L, weight, sca_tri, merged_result
    # return stderr, rho, SNR, fMSE, ecc_values,merged_result
    return stderr, rho, SNR, fMSE, ecc_values

# 加载降水数据
# SM2RAIN = np.load('D:/hongyouting/data/0.25_clip/sm2rain_precipitation.npy')
# ERA5 = np.load('D:/hongyouting/data/0.25_clip/era5_precipitation.npy')
# Satellite = np.load('D:/hongyouting/data/0.25_clip/imerg_e7_4926_precipitation.npy')

# SM2RAIN = np.load('D:/hongyouting/data/0.25_clip_60/sm2rain_precipitation.npy')
# ERA5 = np.load('D:/hongyouting/data/0.25_clip_60/era5_precipitation.npy')
# Satellite = np.load('D:/hongyouting/data/0.25_clip_60/gsmap_nrtg_precipitation.npy')
# Satellite = np.load('D:/hongyouting/data/0.25_clip_60/cmorph_precipitation.npy')
# Satellite = np.load('D:/hongyouting/data/0.25_clip_60/cdr_precipitation.npy')

# SM2RAIN = np.load('D:/hongyouting/data/0.25_clip_60/1/sm2rain_precipitation.npy')
# ERA5 = np.load('D:/hongyouting/data/0.25_clip_60/1/era5_precipitation.npy')
# Satellite = np.load('D:/hongyouting/data/0.25_clip_60/gsmap_precipitation.npy')

SM2RAIN = np.load('D:/hongyouting/data/0.25_clip_50/4930/sm2rain_precipitation.npz')['array']
# ERA5 = np.load('D:/hongyouting/data/0.25_clip_50/4930/era5_precipitation.npz') ['array']
Satellite1 = np.load('D:/hongyouting/data/0.25_clip_50/4930/cdr_precipitation.npz')['array']
Satellite2 = np.load('D:/hongyouting/data/0.25_clip_50/493d0/imerg_f7_precipitation.npz')['array']

# SM2RAIN = np.load('D:/hongyouting/data/0.25_clip_50/4930/sm2rain_precipitation.npz')['array']
# # ERA5 = np.load('D:/hongyouting/data/0.25_clip_50/era5_precipitation.npz') ['array']
# Satellite1 = np.load('D:/hongyouting/data/0.25_clip_50/4930/gsmap_nrtg_precipitation.npz')['array']
# Satellite2 = np.load('D:/hongyouting/data/0.25_clip_50/4930/cmorph_precipitation.npz')['array']

# 在全局命名空间中找到这些数组的变量名

# 开始记录程序运行时间
start_time = time.time()
# for i in range(720):
# for i in range(480):
for i in range(400):
    for j in range(1440):
        try:
            # 获取每个数据集的值
            x = SM2RAIN[:, i, j]
            y = Satellite1[:, i, j]
            z = Satellite2[:, i, j]
            # 如果值都不是NaN，则进行计算
            if np.all(~np.isnan(x)) and np.all(~np.isnan(y)) and np.all(~np.isnan(z)):
                # 构造三元组
                tri = np.column_stack((x, y, z))
                # 计算标准误，相关系数，信噪比，fMSE, ECC
                stderr, rho, SNR, fMSE, ecc_values = EIVD(tri)
                print(i)
                print(j)
                print(stderr)
                print(rho)
                print(SNR)
                print(fMSE)
                print(ecc_values)
                # 构造DataFrame
                # result = pd.DataFrame({'row': i, 'column': j,
                #                        name[0] + '_stderr': stderr[0], name[0] + '_rho': rho[0],
                #                        name[0] + '_snr': SNR[0], name[0] + '_fmse': fMSE[0],
                #                        name[1] + '_stderr': stderr[1], name[1] + '_rho': rho[1],
                #                        name[1] + '_snr': SNR[1], name[1] + '_fmse': fMSE[1],
                #                        name[2] + '_stderr': stderr[2], name[2] + '_rho': rho[2],
                #                        name[2] + '_snr': SNR[2], name[2] + '_fmse': fMSE[2],
                #                        'ecc_1_2': ecc_values[0], 'ecc_1_3': ecc_values[1], 'ecc_2_3': ecc_values[2]},
                #                       index=range(1))
                result = pd.DataFrame({
                    # 'row': [i], 'column': [j],
                    # 'SM2RAIN_stderr': [stderr[0]], 'SM2RAIN_rho': [rho[0]], 'SM2RAIN_snr': [SNR[0]], 'SM2RAIN_fmse': [fMSE[0]],
                    # 'ERA5_stderr': [stderr[1]], 'ERA5_rho': [rho[1]], 'ERA5_snr': [SNR[1]], 'ERA5_fmse': [fMSE[1]],
                    # 'Satellite_stderr': [stderr[2]], 'Satellite_rho': [rho[2]], 'Satellite_snr': [SNR[2]], 'Satellite_fmse': [fMSE[2]],
                    # 'ecc': [ecc_values],
                    # 'merged_result':[merged_result]
                    'row': [i], 'column': [j],
                    'SM2RAIN_stderr': [stderr[0]], 'SM2RAIN_rho': [rho[0]], 'SM2RAIN_snr': [SNR[0]],'SM2RAIN_fmse': [fMSE[0]],
                    'sate1_stderr': [stderr[1]], 'sate1_rho': [rho[1]], 'sate1_snr': [SNR[1]], 'sate1_fmse': [fMSE[1]],
                    'sate2_stderr': [stderr[2]], 'sate2_rho': [rho[2]], 'sate2_snr': [SNR[2]],'sate2_fmse': [fMSE[2]],
                    'ecc': [ecc_values],
                })

                # if not os.path.isfile('D:/hongyouting/result/EIVDresults/global1/esimerg_e7_EIVD_result.csv'):
                #     # 如果文件不存在，则创建一个新的DataFrame，并将其保存到CSV文件中
                #     result.to_csv('D:/hongyouting/result/EIVDresults/global1/esimerg_e7_EIVD_result.csv', index=False)
                # else:
                #     # 如果文件存在，则读取现有的CSV文件，并将新的数据追加到CSV文件中
                #     result.to_csv('D:/hongyouting/result/EIVDresults/global1/esimerg_e7_EIVD_result.csv', mode='a',
                #                   header=False,index=False)

                # if not os.path.isfile('D:/hongyouting/result/EIVDresults/global1/escmorph_EIVD_result.csv'):
                #     # 如果文件不存在，则创建一个新的DataFrame，并将其保存到CSV文件中
                #     result.to_csv('D:/hongyouting/result/EIVDresults/global1/escmorph_EIVD_result.csv', index=False)
                # else:
                #     # 如果文件存在，则读取现有的CSV文件，并将新的数据追加到CSV文件中
                #     result.to_csv('D:/hongyouting/result/EIVDresults/global1/escmorph_EIVD_result.csv', mode='a', header=False,index=False)

                # if not os.path.isfile('D:/hongyouting/result/EIVDresults/global1/escdr_EIVD_result.csv'):
                #     # 如果文件不存在，则创建一个新的DataFrame，并将其保存到CSV文件中
                #     result.to_csv('D:/hongyouting/result/EIVDresults/global1/escdr_EIVD_result.csv', index=False)
                # else:
                #     # 如果文件存在，则读取现有的CSV文件，并将新的数据追加到CSV文件中
                #     result.to_csv('D:/hongyouting/result/EIVDresults/global1/escdr_EIVD_result.csv', mode='a', header=False,index=False)

                # if not os.path.isfile('D:/hongyouting/result/EIVDresults/global1/esgsmap_EIVD_result.csv'):
                #     # 如果文件不存在，则创建一个新的DataFrame，并将其保存到CSV文件中
                #     result.to_csv('D:/hongyouting/result/EIVDresults/global1/esgsmap_EIVD_result.csv', index=False)
                # else:
                #     # 如果文件存在，则读取现有的CSV文件，并将新的数据追加到CSV文件中
                #     result.to_csv('D:/hongyouting/result/EIVDresults/global1/esgsmap_EIVD_result.csv', mode='a', header=False,index=False)

                if not os.path.isfile('D:/hongyouting/result/EIVDresults/global2/sm/cdr_f7_EIVD_result.csv'):
                    # 如果文件不存在，则创建一个新的DataFrame，并将其保存到CSV文件中
                    result.to_csv('D:/hongyouting/result/EIVDresults/global2/sm/cdr_f7_EIVD_result.csv', index=False)
                else:
                    # 如果文件存在，则读取现有的CSV文件，并将新的数据追加到CSV文件中
                    result.to_csv('D:/hongyouting/result/EIVDresults/global2/sm/cdr_f7_EIVD_result.csv', mode='a', header=False,index=False)

                # if not os.path.isfile('D:/hongyouting/result/EIVDresults/gsmap_nrt/esgsmap_g_EIVD_result.csv'):
                #     # 如果文件不存在，则创建一个新的DataFrame，并将其保存到CSV文件中
                #     result.to_csv('D:/hongyouting/result/EIVDresults/gsmap_nrt/esgsmap_g_EIVD_result.csv', index=False)
                # else:
                #     # 如果文件存在，则读取现有的CSV文件，并将新的数据追加到CSV文件中
                #     result.to_csv('D:/hongyouting/result/EIVDresults/gsmap_nrt/esgsmap_g_EIVD_result.csv', mode='a', header=False,index=False)



        except Exception as e:
            print(i)
            print(j)
            error = pd.DataFrame({'row': i, 'column': j}, index=range(1))
            # if not os.path.isfile('D:/hongyouting/result/EIVDresults/global1/esimerg_e7_EIVD_error.csv'):
            #     # 如果文件不存在，则创建一个新的DataFrame，并将其保存到CSV文件中
            #     error.to_csv('D:/hongyouting/result/EIVDresults/global1/esimerg_e7_EIVD_error.csv', index=False)
            # else:
            #     # 如果文件存在，则读取现有的CSV文件，并将新的数据追加到CSV文件中
            #     error.to_csv('D:/hongyouting/result/EIVDresults/global1/esimerg_e7_EIVD_error.csv', mode='a', header=False,
            #                  index=False)

            # if not os.path.isfile('D:/hongyouting/result/EIVDresults/global1/escmorph_EIVD_error.csv'):
            #     # 如果文件不存在，则创建一个新的DataFrame，并将其保存到CSV文件中
            #     error.to_csv('D:/hongyouting/result/EIVDresults/global1/escmorph_EIVD_error.csv', index=False)
            # else:
            #     # 如果文件存在，则读取现有的CSV文件，并将新的数据追加到CSV文件中
            #     error.to_csv('D:/hongyouting/result/EIVDresults/global1/escmorph_EIVD_error.csv', mode='a', header=False,index=False)

            # if not os.path.isfile('D:/hongyouting/result/EIVDresults/global1/escdr_EIVD_error.csv'):
            #     # 如果文件不存在，则创建一个新的DataFrame，并将其保存到CSV文件中
            #     error.to_csv('D:/hongyouting/result/EIVDresults/global1/escdr_EIVD_error.csv', index=False)
            # else:
            #     # 如果文件存在，则读取现有的CSV文件，并将新的数据追加到CSV文件中
            #     error.to_csv('D:/hongyouting/result/EIVDresults/global1/escdr_EIVD_error.csv', mode='a', header=False,index=False)

            # if not os.path.isfile('D:/hongyouting/result/EIVDresults/global1/esgsmap_EIVD_error.csv'):
            #     # 如果文件不存在，则创建一个新的DataFrame，并将其保存到CSV文件中
            #     error.to_csv('D:/hongyouting/result/EIVDresults/global1/esgsmap_EIVD_error.csv', index=False)
            # else:
            #     # 如果文件存在，则读取现有的CSV文件，并将新的数据追加到CSV文件中
            #     error.to_csv('D:/hongyouting/result/EIVDresults/global1/esgsmap_EIVD_error.csv', mode='a',header=False, index=False)

            # if not os.path.isfile('D:/hongyouting/result/EIVDresults/test/eschirps_EIVD_error.csv'):
            #     # 如果文件不存在，则创建一个新的DataFrame，并将其保存到CSV文件中
            #     error.to_csv('D:/hongyouting/result/EIVDresults/test/eschirps_EIVD_error.csv', index=False)
            # else:
            #     # 如果文件存在，则读取现有的CSV文件，并将新的数据追加到CSV文件中
            #     error.to_csv('D:/hongyouting/result/EIVDresults/test/eschirps_EIVD_error.csv', mode='a',header=False, index=False)

            # if not os.path.isfile('D:/hongyouting/result/EIVDresults/gsmap_nrt/esgsmap_g_EIVD_error.csv'):
            #     # 如果文件不存在，则创建一个新的DataFrame，并将其保存到CSV文件中
            #     error.to_csv('D:/hongyouting/result/EIVDresults/gsmap_nrt/esgsmap_g_EIVD_error.csv', index=False)
            # else:
            #     # 如果文件存在，则读取现有的CSV文件，并将新的数据追加到CSV文件中
            #     error.to_csv('D:/hongyouting/result/EIVDresults/gsmap_nrt/esgsmap_g_EIVD_error.csv', mode='a',header=False, index=False)


            print("发生 TypeError:", e)

# 结束记录程序运行时间
end_time = time.time()
total_time = end_time - start_time

print(f"程序运行总时间: {total_time} 秒")
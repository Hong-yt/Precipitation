import numpy as np
import pandas as pd
import os
import time


def EIVD(tri):
    if tri.shape[0] < 4:
        stderr = np.full(3, np.nan)
        rho2 = np.full(3, np.nan)
        SNR = np.full(3, np.nan)
        fMSE = np.full(3, np.nan)
        ecc_values = np.full(1, np.nan)
        return stderr, rho2, SNR, fMSE, ecc_values

        # weight = np.full((1, 3), np.nan)
        # sca_tri = np.full((1, 3), np.nan)
        # return weight, sca_tri

    ExxT_unres = np.cov(tri, rowvar=False)
    sca_tri = np.zeros_like(tri)
    sca_tri[:, 0] = tri[:, 0]

    # sca_tri[:, 1] = (ExxT_unres[0, 2] / ExxT_unres[1, 2]) * (tri[:, 1] - np.nanmean(tri[:, 1])) + np.nanmean(tri[:, 0])
    # sca_tri[:, 2] = (ExxT_unres[0, 1] / ExxT_unres[2, 1]) * (tri[:, 2] - np.nanmean(tri[:, 2])) + np.nanmean(tri[:, 0])
    # 检查分母是否为零
    if ExxT_unres[1, 2] != 0:
        sca_tri[:, 1] = (ExxT_unres[0, 2] / ExxT_unres[1, 2]) * (tri[:, 1] - np.nanmean(tri[:, 1])) + np.nanmean(
            tri[:, 0])
    else:
        sca_tri[:, 1] = np.nan  # 或者选择其他适当的处理方式

    if ExxT_unres[2, 1] != 0:
        sca_tri[:, 2] = (ExxT_unres[0, 1] / ExxT_unres[2, 1]) * (tri[:, 2] - np.nanmean(tri[:, 2])) + np.nanmean(
            tri[:, 0])
    else:
        sca_tri[:, 2] = np.nan  # 或者选择其他适当的处理方式

    ExxT = np.cov(sca_tri, rowvar=False)
    L_tri = sca_tri.copy()
    tri_lag = L_tri[1:, :]
    L_tri = L_tri[:-1, :]
    # L = np.nanmean(L_tri * tri_lag, axis=0)

    # 检查 L_tri 和 tri_lag 是否包含有效数据
    if not np.isnan(L_tri).all() and not np.isnan(tri_lag).all():
        L_tri_times_tri_lag = L_tri * tri_lag
        if not np.isnan(L_tri_times_tri_lag).all():
            L = np.nanmean(L_tri_times_tri_lag, axis=0)
        else:
            L = np.full(L_tri.shape[1], np.nan)
    else:
        L = np.full(L_tri.shape[1], np.nan)

    if np.isnan(L).all():
        stderr = np.full(3, np.nan)
        rho2 = np.full(3, np.nan)
        SNR = np.full(3, np.nan)
        fMSE = np.full(3, np.nan)
        ecc_values = np.full(1, np.nan)
        return stderr, rho2, SNR, fMSE, ecc_values

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

    ecc_values = x[7] / np.sqrt(x[5] * x[6])

    # eta = np.ones((EeeT.shape[1], 1))
    # weight = np.linalg.solve(EeeT, eta) / (eta.T @ (np.linalg.solve(EeeT, eta)))

    # merged_result = np.sum(sca_tri * weight.T, axis=1)

    return stderr, rho, SNR, fMSE, ecc_values

# 加载数据
# MSWEP = np.load('D:/hongyouting/data/0.25_clip/mswep_precipitation.npy')
# SM2RAIN = np.load('D:/hongyouting/data/0.25_clip/sm2rain_precipitation.npy')
# ERA5 = np.load('D:/hongyouting/data/0.25_clip/era5_precipitation.npy')
# Satellite = np.load('D:/hongyouting/data/0.25_clip/imerg_e7_4926_precipitation.npy')

# MSWEP = np.load('D:/hongyouting/data/0.25_clip_60/mswep_precipitation.npy')
# SM2RAIN = np.load('D:/hongyouting/data/0.25_clip_60/sm2rain_precipitation.npy')
# ERA5 = np.load('D:/hongyouting/data/0.25_clip_60/era5_precipitation.npy')
# Satellite = np.load('D:/hongyouting/data/0.25_clip_60/cmorph_precipitation.npy')
# Satellite = np.load('D:/hongyouting/data/0.25_clip_60/cdr_precipitation.npy')
# Satellite = np.load('D:/hongyouting/data/0.25_clip_60/gsmap_nrt_precipitation.npy')

# MSWEP = np.load('D:/hongyouting/data/0.25_clip_60/1/mswep_precipitation.npy')
# SM2RAIN = np.load('D:/hongyouting/data/0.25_clip_60/1/sm2rain_precipitation.npy')
# ERA5 = np.load('D:/hongyouting/data/0.25_clip_60/1/era5_precipitation.npy')
# Satellite = np.load('D:/hongyouting/data/0.25_clip_60/gsmap_precipitation.npy')

MSWEP = np.load('D:/hongyouting/data/0.25_clip_50/4930/mswep_precipitation.npz')['array']
# SM2RAIN = np.load('D:/hongyouting/data/0.25_clip_50/4930/sm2rain_precipitation.npz')['array']
ERA5 = np.load('D:/hongyouting/data/0.25_clip_50/4930/era5_precipitation.npz') ['array']
Satellite1 = np.load('D:/hongyouting/data/0.25_clip_50/4930/gsmap_nrtg_precipitation.npz')['array']
Satellite2 = np.load('D:/hongyouting/data/0.25_clip_50/4930/imerg_f7_precipitation.npz')['array']


# 开始记录程序运行时间
start_time = time.time()
rain_categories = {'light': (0, 10), 'moderate': (10, 25), 'heavy': (25, np.inf)}

for i in range(MSWEP.shape[1]):
    for j in range(MSWEP.shape[2]):
        for category, (lower, upper) in rain_categories.items():
            try:
                mask = (MSWEP[:, i, j] >= lower) & (MSWEP[:, i, j] < upper)
                dates = np.where(mask)[0]
                if dates.size == 0:
                    continue

                tri = np.column_stack((ERA5[dates, i, j], Satellite1[dates, i, j], Satellite2[dates, i, j]))
                if np.any(np.isnan(tri)):
                    continue

                if tri.shape[0] < 3:  # 数据不足三组
                    print(f"Skipping Grid ({i}, {j})")
                    continue  # 跳过当前网格点的计算

                stderr, rho, SNR, fMSE, ecc_values = EIVD(tri)
                print(f'Grid ({i}, {j}), Category: {category}')
                print('Standard Error:', stderr)
                print('Correlation Coefficient:', rho)
                print('SNR:', SNR)
                print('fMSE:', fMSE)
                print('ECC Values:', ecc_values)

                result = pd.DataFrame({
                    'row': [i], 'column': [j], 'category': [category],
                    'SM2RAIN_stderr': [stderr[0]], 'SM2RAIN_rho': [rho[0]], 'SM2RAIN_snr': [SNR[0]],'SM2RAIN_fmse': [fMSE[0]],
                    'Sate1_stderr': [stderr[1]], 'Sate1_rho': [rho[1]], 'Sate1_snr': [SNR[1]], 'Sate1_fmse': [fMSE[1]],
                    'Sate2_stderr': [stderr[2]], 'Sate2_rho': [rho[2]], 'Sate2_snr': [SNR[2]],'Sate2_fmse': [fMSE[2]],
                    'ecc': [ecc_values],
                })

                filename = f'D:/hongyouting/result/EIVDresults/categorization2/era5/nrtg_f7_EIVD_{category}.csv'
                # filename = f'D:/hongyouting/result/EIVDresults/categorization2/esgsmap_EIVD_{category}.csv'
                # filename = f'D:/hongyouting/result/EIVDresults/categorization2/escmorph_EIVD_{category}.csv'
                # filename = f'D:/hongyouting/result/EIVDresults/categorization2/escdr_EIVD_{category}.csv'
                # filename = f'D:/hongyouting/result/EIVDresults/categorization2/eschirps_EIVD_{category}.csv'

                # filename = f'D:/hongyouting/result/EIVDresults/gsmap_nrt/esgsmap_EIVD_{category}.csv'


                if not os.path.isfile(filename):
                    result.to_csv(filename, index=False)
                else:
                    result.to_csv(filename, mode='a', header=False, index=False)

            except Exception as e:
                print(f'Error at Grid ({i}, {j}), Category: {category}:', e)
                # error = pd.DataFrame({'row': [i], 'column': [j], 'category': [category]})
                # error_filename = f'D:/hongyouting/result/EIVDresults/categorization2/sm/nrtg_f7_{category}_EIVD_error.csv'
                # error_filename = f'D:/hongyouting/result/EIVDresults/categorization2/esgsmap_{category}_EIVD_error.csv'
                # error_filename = f'D:/hongyouting/result/EIVDresults/categorization2/escmorph_{category}_EIVD_error.csv'
                # error_filename = f'D:/hongyouting/result/EIVDresults/categorization2/escdr_{category}_EIVD_error.csv'
                # error_filename = f'D:/hongyouting/result/EIVDresults/categorization2/eschirps_{category}_EIVD_error.csv'

                # error_filename = f'D:/hongyouting/result/EIVDresults/gsmap_nrt/esgsmap_{category}_EIVD_error.csv'


                # if not os.path.isfile(error_filename):
                #     error.to_csv(error_filename, index=False)
                # else:
                #     error.to_csv(error_filename, mode='a', header=False, index=False)

end_time = time.time()
total_time = end_time - start_time

print(f"程序运行总时间: {total_time} 秒")

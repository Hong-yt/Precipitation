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

    # 协方差矩阵
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

    x = np.linalg.inv(A.T @ A) @ A.T @ y  # 通过最小二乘法求解线性方程组的解

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
    fMSE = 1 - rho2
    rho = np.sqrt(rho2)

    ecc_values = x[7] / np.sqrt(x[5] * x[6])

    return stderr, rho, SNR, fMSE, ecc_values


# 数据加载（添加调试信息）
print("开始加载数据...")

try:
    SM2RAIN = np.load('D:/hongyouting/data/0.25_clip_50/4930/sm2rain_precipitation.npz')['array']
    print(f"SM2RAIN 数据形状: {SM2RAIN.shape}")

    Satellite1 = np.load('D:/hongyouting/data/0.25_clip_50/4930/cdr_precipitation.npz')['array']
    print(f"Satellite1 数据形状: {Satellite1.shape}")

    Satellite2 = np.load('D:/hongyouting/data/0.25_clip_50/4930/imerg_f7_precipitation.npz')['array']
    # Satellite2 = np.load('D:/hongyouting/data/0.25_clip_50/4930/imerg_e7_precipitation.npz')['array']

    print(f"Satellite2 数据形状: {Satellite2.shape}")

    print("数据加载完成！")

except Exception as e:
    print(f"数据加载失败: {e}")
    exit()

# 创建输出目录（如果不存在）
output_dir = 'D:/hongyouting/result/EIVDresults/global2/sm/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"创建输出目录: {output_dir}")

# 开始记录程序运行时间
start_time = time.time()
processed_count = 0
error_count = 0

print("开始处理数据...")
print("开始进入主循环...")
for i in range(400):
    for j in range(1440):
        if i % 100 == 0 and j % 500 == 0:
            print(f"处理点 ({i},{j})")
        try:
            # 获取每个数据集的值
            x = SM2RAIN[:, i, j]
            y = Satellite1[:, i, j]
            z = Satellite2[:, i, j]

            # 检查数据有效性（修改条件，允许部分NaN值）
            valid_x = ~np.isnan(x)
            valid_y = ~np.isnan(y)
            valid_z = ~np.isnan(z)

            # 至少需要一定数量的有效数据点
            min_valid_points = max(4, len(x) * 0.7)  # 至少70%的数据有效
            valid_count = np.sum(valid_x & valid_y & valid_z)

            if valid_count >= min_valid_points:
                # 构造三元组，只使用有效数据点
                valid_indices = valid_x & valid_y & valid_z
                tri = np.column_stack((x[valid_indices], y[valid_indices], z[valid_indices]))

                # 计算EIVD
                stderr, rho, SNR, fMSE, ecc_values = EIVD(tri)

                # 每处理100个点显示一次进度
                if processed_count % 1440 == 0:
                    print(f"处理进度: i={i}, j={j}, 已处理: {processed_count}")

                # 构造DataFrame
                result = pd.DataFrame({
                    'row': [i], 'column': [j],
                    'SM2RAIN_stderr': [stderr[0]], 'SM2RAIN_rho': [rho[0]], 'SM2RAIN_snr': [SNR[0]],
                    'SM2RAIN_fmse': [fMSE[0]],
                    'sate1_stderr': [stderr[1]], 'sate1_rho': [rho[1]], 'sate1_snr': [SNR[1]], 'sate1_fmse': [fMSE[1]],
                    'sate2_stderr': [stderr[2]], 'sate2_rho': [rho[2]], 'sate2_snr': [SNR[2]], 'sate2_fmse': [fMSE[2]],
                    'ecc': [ecc_values],
                })

                # 保存结果
                output_file = os.path.join(output_dir, 'cdr_f7_EIVD_result.csv')
                output_file = os.path.join(output_dir, 'cdr_e7_EIVD_result.csv')

                if not os.path.isfile(output_file):
                    result.to_csv(output_file, index=False)
                    print(f"创建新文件: {output_file}")
                else:
                    result.to_csv(output_file, mode='a', header=False, index=False)

                processed_count += 1

        except Exception as e:
            error_count += 1
            # print(f"处理错误 - 位置({i}, {j}): {str(e)}")

            # # 保存错误信息
            # error = pd.DataFrame({'row': [i], 'column': [j], 'error': [str(e)]})
            # error_file = os.path.join(output_dir, 'cdr_f7_EIVD_error.csv')
            # error_file = os.path.join(output_dir, 'cdr_e7_EIVD_error.csv')

            # if not os.path.isfile(error_file):
            #     error.to_csv(error_file, index=False)
            # else:
            #     error.to_csv(error_file, mode='a', header=False, index=False)

# 结束记录程序运行时间
end_time = time.time()
total_time = end_time - start_time

print(f"\n程序运行完成!")
print(f"总运行时间: {total_time:.2f} 秒")
print(f"成功处理: {processed_count} 个数据点")
print(f"错误数量: {error_count} 个")
print(f"处理速度: {processed_count / total_time:.2f} 点/秒")

# 检查输出文件
output_file = os.path.join(output_dir, 'cdr_f7_EIVD_result.csv')
# output_file = os.path.join(output_dir, 'cdr_e7_EIVD_result.csv')

if os.path.exists(output_file):
    file_size = os.path.getsize(output_file)
    print(f"输出文件大小: {file_size} 字节")

    # 读取并显示前几行结果
    try:
        df = pd.read_csv(output_file)
        print(f"结果文件包含 {len(df)} 行数据")
        print("前5行数据:")
        print(df.head())
    except Exception as e:
        print(f"读取结果文件失败: {e}")
else:
    print("警告: 没有生成输出文件!")
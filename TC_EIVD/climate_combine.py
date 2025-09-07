# ——————————————————————————————station——————————————————————————
# import pandas as pd
# import os
# from pathlib import Path
#
#
# def process_folders():
#     """
#     处理8个文件夹中的CSV文件，按土地利用类型分组汇总
#     """
#
#     # 定义文件夹名称
#     folders = [
#         'D:/hongyouting/result/climate_results/STATION/imerg_f7',
#         'D:/hongyouting/result/climate_results/STATION/imerg_e7',
#         'D:/hongyouting/result/climate_results/STATION/gsmap_mvkg',
#         'D:/hongyouting/result/climate_results/STATION/gsmap_nrt',
#         'D:/hongyouting/result/climate_results/STATION/gsmap_nrtg',
#         'D:/hongyouting/result/climate_results/STATION/cmorph',
#         'D:/hongyouting/result/climate_results/STATION/chirps',
#         'D:/hongyouting/result/climate_results/STATION/cdr'
#     ]
#
#     # 定义土地利用类型映射
#     climate_mapping = {
#         'A': ['Af', 'Am', 'Aw'],
#         'B': ['Bs', 'Bw'],
#         'C': ['Cf', 'Cs','Cw'],
#         'D': ['Df','Ds','Dw'],
#         'E': ['EF','ET'],
#     }
#
#     # 创建combine文件夹
#     combine_folder = Path('combine')
#     combine_folder.mkdir(exist_ok=True)
#
#     for folder_name in folders:
#         print(f"正在处理文件夹: {folder_name}")
#
#         folder_path = Path(folder_name)
#         if not folder_path.exists():
#             print(f"警告: 文件夹 {folder_name} 不存在，跳过处理")
#             continue
#
#         # 创建combine子文件夹
#         combine_subfolder = folder_path / 'combine'
#         combine_subfolder.mkdir(exist_ok=True)
#
#         # 第一步：按类型分组汇总
#         type_dataframes = {}
#
#         for type_name, type_numbers in climate_mapping.items():
#             combined_data = []
#
#             for type_num in type_numbers:
#                 csv_file = folder_path / f'climate_{type_num}.csv'
#
#                 if csv_file.exists():
#                     try:
#                         df = pd.read_csv(csv_file)
#                         # 只保留需要的字段
#                         if all(col in df.columns for col in
#                                ['station_id', 'latitude', 'longitude', 'r', 'climate_type']):
#                             df_selected = df[['station_id', 'latitude', 'longitude', 'r', 'climate_type']].copy()
#                             combined_data.append(df_selected)
#                         else:
#                             print(f"警告: {csv_file} 缺少必要的列")
#                     except Exception as e:
#                         print(f"错误: 读取 {csv_file} 时出错: {e}")
#                 else:
#                     print(f"警告: 文件 {csv_file} 不存在")
#
#             if combined_data:
#                 # 合并数据
#                 type_df = pd.concat(combined_data, ignore_index=True)
#                 type_dataframes[type_name] = type_df
#
#                 # 保存第一步的结果到combine子文件夹
#                 output_file = combine_subfolder / f'{type_name}.csv'
#                 type_df.to_csv(output_file, index=False)
#                 print(f"  已保存: {output_file}")
#
#         # 第二步：进一步汇总，只保留station_id和各类型的r值
#         if type_dataframes:
#             # 找到所有唯一的station_id
#             all_station_ids = set()
#             for df in type_dataframes.values():
#                 station_ids = set(df['station_id'].astype(str))  # 转换为字符串统一处理
#                 all_station_ids.update(station_ids)
#
#             # 转换为列表并排序（现在都是字符串，可以正常排序）
#             all_station_ids = sorted(list(all_station_ids))
#
#             # 创建基础数据框
#             final_df = pd.DataFrame(all_station_ids, columns=['station_id'])
#
#             # 为每个类型添加r列
#             for type_name, type_df in type_dataframes.items():
#                 # 确保station_id为字符串类型
#                 type_df_copy = type_df.copy()
#                 type_df_copy['station_id'] = type_df_copy['station_id'].astype(str)
#
#                 # 按station_id分组，对r求平均值
#                 grouped = type_df_copy.groupby(['station_id'])['r'].mean().reset_index()
#
#                 # 合并到最终数据框
#                 final_df = final_df.merge(
#                     grouped.rename(columns={'r': type_name}),
#                     on=['station_id'],
#                     how='left'
#                 )
#
#             # 填充NaN值为0（如果需要的话，可以取消注释）
#             # for type_name in climate_mapping.keys():
#             #     if type_name in final_df.columns:
#             #         final_df[type_name] = final_df[type_name].fillna(0)
#
#             # 获取文件夹名称（最后一部分）
#             folder_basename = Path(folder_name).name
#
#             # 保存最终汇总结果到combine子文件夹
#             final_output_file = combine_subfolder / f'{folder_basename}_combine.csv'
#             final_df.to_csv(final_output_file, index=False)
#             print(f"  已保存最终汇总: {final_output_file}")
#
#         print(f"完成处理文件夹: {folder_name}\n")
#
#     print("所有文件夹处理完成！")
#
#
# def check_folder_structure():
#     """
#     检查文件夹结构和文件存在情况
#     """
#     folders = [
#         'D:/hongyouting/result/climate_results/STATION/imerg_f7',
#         'D:/hongyouting/result/climate_results/STATION/imerg_e7',
#         'D:/hongyouting/result/climate_results/STATION/gsmap_mvkg',
#         'D:/hongyouting/result/climate_results/STATION/gsmap_nrt',
#         'D:/hongyouting/result/climate_results/STATION/gsmap_nrtg',
#         'D:/hongyouting/result/climate_results/STATION/cmorph',
#         'D:/hongyouting/result/climate_results/STATION/chirps',
#         'D:/hongyouting/result/climate_results/STATION/cdr'
#     ]
#
#     print("检查文件夹结构:")
#     for folder_name in folders:
#         folder_path = Path(folder_name)
#         if folder_path.exists():
#             csv_files = list(folder_path.glob('climate_type_*.csv'))
#             print(f"  {folder_name}: 找到 {len(csv_files)} 个CSV文件")
#
#             # 检查特定类型的文件
#             # required_types = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16]
#             # missing_types = []
#             # for type_num in required_types:
#             #     if not (folder_path / f'climate_type_{type_num}.csv').exists():
#             #         missing_types.append(type_num)
#
#         #     if missing_types:
#         #         print(f"    缺少类型: {missing_types}")
#         # else:
#         #     print(f"  {folder_name}: 文件夹不存在")
#     print()
#
#
# def main():
#     """
#     主函数
#     """
#     print("CSV文件处理和汇总程序")
#     print("=" * 50)
#
#     # 检查文件夹结构
#     check_folder_structure()
#
#     # # 确认是否继续处理
#     # response = input("是否继续处理文件？(y/n): ")
#     # if response.lower() != 'y':
#     #     print("程序已取消")
#     #     return
#
#     # 开始处理
#     try:
#         process_folders()
#         print("\n处理完成！")
#         print("结果说明:")
#         print("1. 每个原文件夹中的combine子文件夹会生成按类型汇总的CSV文件")
#         print("2. 每个原文件夹中的combine子文件夹会生成最终汇总文件（如imerg_f7_combine.csv）")
#
#     except Exception as e:
#         print(f"处理过程中出现错误: {e}")
#         import traceback
#         traceback.print_exc()
#
#
# if __name__ == "__main__":
#     main()
# ——————————————————————————————————mtc——————————————————————————————————
# import pandas as pd
# import os
# from pathlib import Path
#
#
# def process_folders():
#     """
#     处理8个文件夹中的CSV文件，按土地利用类型分组汇总
#     """
#
#     # 定义文件夹名称
#     folders = [
#         'D:/hongyouting/result/climate_results/MTC/imerg_f7',
#         'D:/hongyouting/result/climate_results/MTC/imerg_e7',
#         'D:/hongyouting/result/climate_results/MTC/gsmap_mvkg',
#         'D:/hongyouting/result/climate_results/MTC/gsmap_nrt',
#         'D:/hongyouting/result/climate_results/MTC/gsmap_nrtg',
#         'D:/hongyouting/result/climate_results/MTC/cmorph',
#         'D:/hongyouting/result/climate_results/MTC/chirps',
#         'D:/hongyouting/result/climate_results/MTC/cdr'
#     ]
#
#     # 定义土地利用类型映射
#     climate_mapping = {
#         'A': ['Af', 'Am', 'Aw'],
#         'B': ['Bs', 'Bw'],
#         'C': ['Cf', 'Cs','Cw'],
#         'D': ['Df','Ds','Dw'],
#         'E': ['EF','ET'],
#     }
#
#     # 创建combine文件夹
#     combine_folder = Path('combine')
#     combine_folder.mkdir(exist_ok=True)
#
#     for folder_name in folders:
#         print(f"正在处理文件夹: {folder_name}")
#
#         folder_path = Path(folder_name)
#         if not folder_path.exists():
#             print(f"警告: 文件夹 {folder_name} 不存在，跳过处理")
#             continue
#
#         # 创建combine子文件夹
#         combine_subfolder = folder_path / 'combine'
#         combine_subfolder.mkdir(exist_ok=True)
#
#         # 第一步：按类型分组汇总
#         type_dataframes = {}
#
#         for type_name, type_numbers in climate_mapping.items():
#             combined_data = []
#
#             for type_num in type_numbers:
#                 csv_file = folder_path / f'climate_{type_num}.csv'
#
#                 if csv_file.exists():
#                     try:
#                         df = pd.read_csv(csv_file)
#                         # 只保留需要的字段
#                         if all(col in df.columns for col in
#                                ['i', 'j', 'CC_satellite', 'climate_type']):
#                             df_selected = df[['i', 'j', 'CC_satellite', 'climate_type']].copy()
#                             combined_data.append(df_selected)
#                         else:
#                             print(f"警告: {csv_file} 缺少必要的列")
#                     except Exception as e:
#                         print(f"错误: 读取 {csv_file} 时出错: {e}")
#                 else:
#                     print(f"警告: 文件 {csv_file} 不存在")
#
#             if combined_data:
#                 # 合并数据
#                 type_df = pd.concat(combined_data, ignore_index=True)
#                 type_dataframes[type_name] = type_df
#
#                 # 保存第一步的结果到combine子文件夹
#                 output_file = combine_subfolder / f'{type_name}.csv'
#                 type_df.to_csv(output_file, index=False)
#                 print(f"  已保存: {output_file}")
#
#         # 第二步：进一步汇总，只保留CC_satellite
#         if type_dataframes:
#             # 创建最终的汇总数据框
#             final_data = {}
#
#             # 找到所有唯一的i,j组合
#             all_coords = set()
#             for df in type_dataframes.values():
#                 coords = set(zip(df['i'], df['j']))
#                 all_coords.update(coords)
#
#             # 转换为列表并排序
#             all_coords = sorted(list(all_coords))
#
#             # 创建基础数据框
#             final_df = pd.DataFrame(all_coords, columns=['i', 'j'])
#
#             # 为每个类型添加CC_satellite列
#             for type_name, type_df in type_dataframes.items():
#                 # 按i,j分组，对CC_satellite求和或平均（这里使用平均值）
#                 grouped = type_df.groupby(['i', 'j'])['CC_satellite'].mean().reset_index()
#
#                 # 合并到最终数据框
#                 final_df = final_df.merge(
#                     grouped.rename(columns={'CC_satellite': type_name}),
#                     on=['i', 'j'],
#                     how='left'
#                 )
#
#             # 保持NaN值为空，不填充为0
#             # for type_name in landuse_mapping.keys():
#             #     if type_name in final_df.columns:
#             #         final_df[type_name] = final_df[type_name].fillna(0)
#
#             # 获取文件夹名称（最后一部分）
#             folder_basename = Path(folder_name).name
#
#             # 保存最终汇总结果到combine子文件夹
#             final_output_file = combine_subfolder / f'{folder_basename}_combine.csv'
#             final_df.to_csv(final_output_file, index=False)
#             print(f"  已保存最终汇总: {final_output_file}")
#
#         print(f"完成处理文件夹: {folder_name}\n")
#
#     print("所有文件夹处理完成！")
#
#
# def check_folder_structure():
#     """
#     检查文件夹结构和文件存在情况
#     """
#     folders = [
#         # 'D:/hongyouting/result/climate_results/MTC/imerg_f7',
#         # 'D:/hongyouting/result/climate_results/MTC/imerg_e7',
#         'D:/hongyouting/result/climate_results/MTC/gsmap_mvkg',
#         'D:/hongyouting/result/climate_results/MTC/gsmap_nrt',
#         'D:/hongyouting/result/climate_results/MTC/gsmap_nrtg',
#         'D:/hongyouting/result/climate_results/MTC/cmorph',
#         'D:/hongyouting/result/climate_results/MTC/chirps',
#         'D:/hongyouting/result/climate_results/MTC/cdr'
#     ]
#
#     print("检查文件夹结构:")
#     for folder_name in folders:
#         folder_path = Path(folder_name)
#         if folder_path.exists():
#             csv_files = list(folder_path.glob('climate_*.csv'))
#             print(f"  {folder_name}: 找到 {len(csv_files)} 个CSV文件")
#
#             # 检查特定类型的文件
#             # required_types = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16]
#             # missing_types = []
#             # for type_num in required_types:
#             #     if not (folder_path / f'climate_type_{type_num}.csv').exists():
#             #         missing_types.append(type_num)
#
#         #     if missing_types:
#         #         print(f"    缺少类型: {missing_types}")
#         # else:
#         #     print(f"  {folder_name}: 文件夹不存在")
#     print()
#
#
# def main():
#     """
#     主函数
#     """
#     print("CSV文件处理和汇总程序")
#     print("=" * 50)
#
#     # 检查文件夹结构
#     check_folder_structure()
#
#     # # 确认是否继续处理
#     # response = input("是否继续处理文件？(y/n): ")
#     # if response.lower() != 'y':
#     #     print("程序已取消")
#     #     return
#
#     # 开始处理
#     try:
#         process_folders()
#         print("\n处理完成！")
#         print("结果说明:")
#         print("1. 每个原文件夹中的combine子文件夹会生成按类型汇总的CSV文件")
#         print("2. 每个原文件夹中的combine子文件夹会生成最终汇总文件（如imerg_f7_combine.csv）")
#
#     except Exception as e:
#         print(f"处理过程中出现错误: {e}")
#         import traceback
#         traceback.print_exc()
#
#
# if __name__ == "__main__":
#     main()
# ——————————————————————————————————————————eivd————————————————————————————————————————————
import pandas as pd
import os
from pathlib import Path


def process_folders():
    """
    处理8个文件夹中的CSV文件，按土地利用类型分组汇总
    """

    # 定义文件夹名称
    folders = [
        # 'D:/hongyouting/result/climate_results/EIVD/imerg_f7',
        # 'D:/hongyouting/result/climate_results/EIVD/imerg_e7',
        'D:/hongyouting/result/climate_results/EIVD/gsmap_mvkg',
        'D:/hongyouting/result/climate_results/EIVD/gsmap_nrt',
        'D:/hongyouting/result/climate_results/EIVD/gsmap_nrtg',
        # 'D:/hongyouting/result/climate_results/EIVD/cmorph',
        # 'D:/hongyouting/result/climate_results/EIVD/chirps',
        'D:/hongyouting/result/climate_results/EIVD/cdr'
    ]

    # 定义土地利用类型映射
    climate_mapping = {
        'A': ['Af', 'Am', 'Aw'],
        'B': ['Bs', 'Bw'],
        'C': ['Cf', 'Cs','Cw'],
        'D': ['Df','Ds','Dw'],
        'E': ['EF','ET'],
    }

    # 创建combine文件夹
    combine_folder = Path('combine')
    combine_folder.mkdir(exist_ok=True)

    for folder_name in folders:
        print(f"正在处理文件夹: {folder_name}")

        folder_path = Path(folder_name)
        if not folder_path.exists():
            print(f"警告: 文件夹 {folder_name} 不存在，跳过处理")
            continue

        # 创建combine子文件夹
        combine_subfolder = folder_path / 'combine'
        combine_subfolder.mkdir(exist_ok=True)

        # 第一步：按类型分组汇总
        type_dataframes = {}

        for type_name, type_numbers in climate_mapping.items():
            combined_data = []

            for type_num in type_numbers:
                csv_file = folder_path / f'climate_{type_num}.csv'

                if csv_file.exists():
                    try:
                        df = pd.read_csv(csv_file)
                        # 只保留需要的字段
                        if all(col in df.columns for col in
                               ['row', 'column', 'sate1_rho', 'climate_type']):
                            df_selected = df[['row', 'column', 'sate1_rho', 'climate_type']].copy()
                            combined_data.append(df_selected)
                        else:
                            print(f"警告: {csv_file} 缺少必要的列")
                    except Exception as e:
                        print(f"错误: 读取 {csv_file} 时出错: {e}")
                else:
                    print(f"警告: 文件 {csv_file} 不存在")

            if combined_data:
                # 合并数据
                type_df = pd.concat(combined_data, ignore_index=True)
                type_dataframes[type_name] = type_df

                # 保存第一步的结果到combine子文件夹
                output_file = combine_subfolder / f'{type_name}.csv'
                type_df.to_csv(output_file, index=False)
                print(f"  已保存: {output_file}")

        # 第二步：进一步汇总，只保留CC_satellite
        if type_dataframes:
            # 创建最终的汇总数据框
            final_data = {}

            # 找到所有唯一的i,j组合
            all_coords = set()
            for df in type_dataframes.values():
                coords = set(zip(df['row'], df['column']))
                all_coords.update(coords)

            # 转换为列表并排序
            all_coords = sorted(list(all_coords))

            # 创建基础数据框
            final_df = pd.DataFrame(all_coords, columns=['row', 'column'])

            # 为每个类型添加CC_satellite列
            for type_name, type_df in type_dataframes.items():
                # 按i,j分组，对CC_satellite求和或平均（这里使用平均值）
                grouped = type_df.groupby(['row', 'column'])['sate1_rho'].mean().reset_index()

                # 合并到最终数据框
                final_df = final_df.merge(
                    grouped.rename(columns={'sate1_rho': type_name}),
                    on=['row', 'column'],
                    how='left'
                )

            # 保持NaN值为空，不填充为0
            # for type_name in landuse_mapping.keys():
            #     if type_name in final_df.columns:
            #         final_df[type_name] = final_df[type_name].fillna(0)

            # 获取文件夹名称（最后一部分）
            folder_basename = Path(folder_name).name

            # 保存最终汇总结果到combine子文件夹
            final_output_file = combine_subfolder / f'{folder_basename}_combine.csv'
            final_df.to_csv(final_output_file, index=False)
            print(f"  已保存最终汇总: {final_output_file}")

        print(f"完成处理文件夹: {folder_name}\n")

    print("所有文件夹处理完成！")


def check_folder_structure():
    """
    检查文件夹结构和文件存在情况
    """
    folders = [
        # 'D:/hongyouting/result/climate_results/EIVD/imerg_f7',
        # 'D:/hongyouting/result/climate_results/EIVD/imerg_e7',
        'D:/hongyouting/result/climate_results/EIVD/gsmap_mvkg',
        'D:/hongyouting/result/climate_results/EIVD/gsmap_nrt',
        'D:/hongyouting/result/climate_results/EIVD/gsmap_nrtg',
        # 'D:/hongyouting/result/climate_results/EIVD/cmorph',
        # 'D:/hongyouting/result/climate_results/EIVD/chirps',
        'D:/hongyouting/result/climate_results/EIVD/cdr'
    ]

    print("检查文件夹结构:")
    for folder_name in folders:
        folder_path = Path(folder_name)
        if folder_path.exists():
            csv_files = list(folder_path.glob('climate_*.csv'))
            print(f"  {folder_name}: 找到 {len(csv_files)} 个CSV文件")

            # 检查特定类型的文件
            # required_types = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16]
            # missing_types = []
            # for type_num in required_types:
            #     if not (folder_path / f'climate_type_{type_num}.csv').exists():
            #         missing_types.append(type_num)

        #     if missing_types:
        #         print(f"    缺少类型: {missing_types}")
        # else:
        #     print(f"  {folder_name}: 文件夹不存在")
    print()


def main():
    """
    主函数
    """
    print("CSV文件处理和汇总程序")
    print("=" * 50)

    # 检查文件夹结构
    check_folder_structure()

    # # 确认是否继续处理
    # response = input("是否继续处理文件？(y/n): ")
    # if response.lower() != 'y':
    #     print("程序已取消")
    #     return

    # 开始处理
    try:
        process_folders()
        print("\n处理完成！")
        print("结果说明:")
        print("1. 每个原文件夹中的combine子文件夹会生成按类型汇总的CSV文件")
        print("2. 每个原文件夹中的combine子文件夹会生成最终汇总文件（如imerg_f7_combine.csv）")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
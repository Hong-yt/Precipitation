import pandas as pd
import os
from pathlib import Path


def process_folders():
    """
    处理8个文件夹中的CSV文件，按土地利用类型分组汇总
    """

    # 定义文件夹名称
    folders = [
        # 'D:/hongyouting/result/landuse_results/EIVD/imerg_f7',
        #        'D:/hongyouting/result/landuse_results/EIVD/imerg_e7',
               'D:/hongyouting/result/landuse_results/EIVD/gsmap_mvkg',
               # 'D:/hongyouting/result/landuse_results/EIVD/gsmap_nrt',
               # 'D:/hongyouting/result/landuse_results/EIVD/gsmap_nrtg',
               # 'D:/hongyouting/result/landuse_results/EIVD/cmorph',
               # 'D:/hongyouting/result/landuse_results/EIVD/chirps',
               # 'D:/hongyouting/result/landuse_results/EIVD/cdr'
        ]

    # 定义土地利用类型映射
    landuse_mapping = {
        'Forests': [1, 2, 3, 4, 5],
        'Shrublands': [6, 7],
        'Savannas': [8, 9],
        'Grasslands': [10],
        'Croplands': [12],
        'Barren': [16]
    }

    # 创建combine文件夹
    combine_folder = Path('combine')
    combine_folder.mkdir(exist_ok=True)

    for folder_name in folders:
        print(f"正在处理文件夹: {folder_name}")

        # 创建combine文件夹
        combine_folder = Path('combine')
        combine_folder.mkdir(exist_ok=True)

        folder_path = Path(folder_name)
        if not folder_path.exists():
            print(f"警告: 文件夹 {folder_name} 不存在，跳过处理")
            continue



        # 第一步：按类型分组汇总
        type_dataframes = {}

        for type_name, type_numbers in landuse_mapping.items():
            combined_data = []

            for type_num in type_numbers:
                csv_file = folder_path / f'landuse_type_{type_num}.csv'

                if csv_file.exists():
                    try:
                        df = pd.read_csv(csv_file)
                        # 只保留需要的字段
                        if all(col in df.columns for col in ['i', 'j', 'Satellite_rho', 'landuse_type']):
                            df_selected = df[['i', 'j', 'Satellite_rho', 'landuse_type']].copy()
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

                # 保存第一步的结果到原文件夹
                output_file = folder_path / f'combine/{type_name}.csv'
                type_df.to_csv(output_file, index=False)
                print(f"  已保存: {output_file}")

        # 第二步：进一步汇总，只保留CC_satellite
        if type_dataframes:
            # 创建最终的汇总数据框
            final_data = {}

            # 找到所有唯一的i,j组合
            all_coords = set()
            for df in type_dataframes.values():
                coords = set(zip(df['i'],df['j']))
                all_coords.update(coords)

            # 转换为列表并排序
            all_coords = sorted(list(all_coords))

            # 创建基础数据框
            final_df = pd.DataFrame(all_coords, columns=['i','j'])

            # 为每个类型添加CC_satellite列
            for type_name, type_df in type_dataframes.items():
                # 按i,j分组，对CC_satellite求和或平均（这里使用平均值）
                grouped = type_df.groupby(['i','j'])['Satellite_rho'].mean().reset_index()

                # 合并到最终数据框
                final_df = final_df.merge(
                    grouped.rename(columns={'Satellite_rho': type_name}),
                    on=['i','j'],
                    how='left'
                )

            # # 填充NaN值为0
            # for type_name in landuse_mapping.keys():
            #     if type_name in final_df.columns:
            #         final_df[type_name] = final_df[type_name].fillna(0)

            # 保存最终汇总结果到原文件夹
            final_output_file = folder_path / f'{folder_name}_combine.csv'
            final_df.to_csv(final_output_file, index=False)
            print(f"  已保存最终汇总: {final_output_file}")

        print(f"完成处理文件夹: {folder_name}\n")

    print("所有文件夹处理完成！")


def check_folder_structure():
    """
    检查文件夹结构和文件存在情况
    """
    folders = [
        # 'D:/hongyouting/result/landuse_results/EIVD/imerg_f7',
        #        'D:/hongyouting/result/landuse_results/EIVD/imerg_e7',
               'D:/hongyouting/result/landuse_results/EIVD/gsmap_mvkg',
               # 'D:/hongyouting/result/landuse_results/EIVD/gsmap_nrt',
               # 'D:/hongyouting/result/landuse_results/EIVD/gsmap_nrtg',
               # 'D:/hongyouting/result/landuse_results/EIVD/cmorph',
               # 'D:/hongyouting/result/landuse_results/EIVD/chirps',
               # 'D:/hongyouting/result/landuse_results/EIVD/cdr'
    ]

    print("检查文件夹结构:")
    for folder_name in folders:
        folder_path = Path(folder_name)
        if folder_path.exists():
            csv_files = list(folder_path.glob('landuse_type_*.csv'))
            print(f"  {folder_name}: 找到 {len(csv_files)} 个CSV文件")

            # 检查特定类型的文件
            required_types = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16]
            missing_types = []
            for type_num in required_types:
                if not (folder_path / f'landuse_type_{type_num}.csv').exists():
                    missing_types.append(type_num)

            if missing_types:
                print(f"    缺少类型: {missing_types}")
        else:
            print(f"  {folder_name}: 文件夹不存在")
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
        print("1. 每个原文件夹中会生成按类型汇总的CSV文件（Forests.csv, Shrublands.csv等）")
        print("2. 每个原文件夹中会生成最终汇总文件（如imerg_f7_summary.csv）")
        print("3. combine文件夹中会包含所有最终汇总文件")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
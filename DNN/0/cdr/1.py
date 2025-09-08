#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从多个独立的交叉验证折中收集结果并合并
处理预测结果，计算评估指标，生成可视化图表
修复了字体显示问题
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sys

# 配置字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.default'] = 'regular'


def convert_to_serializable(obj):
    """转换对象为可序列化的形式"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj


def save_experiment_results(results, output_dir, filename):
    """
    保存实验结果为JSON格式

    参数:
    results: 要保存的结果字典
    output_dir: 输出目录
    filename: 文件名（不包含扩展名）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 转换为可序列化对象
    serializable_results = convert_to_serializable(results)

    # 保存为JSON
    json_path = os.path.join(output_dir, f"{filename}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)

    print(f"实验结果已保存到: {json_path}")


def save_cv_results(all_predictions, all_actuals, cv_metrics, fold_metrics, output_dir, dataset_name):
    """
    保存交叉验证的所有结果

    参数:
    all_predictions: 所有折的预测值集合
    all_actuals: 所有折的实际值集合
    cv_metrics: 每折的评估指标
    fold_metrics: 每折详细的指标
    output_dir: 输出目录
    dataset_name: 数据集名称
    """
    os.makedirs(output_dir, exist_ok=True)

    # 确保转换为NumPy数组
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)

    print(f"预测值形状: {all_predictions.shape}")
    print(f"实际值形状: {all_actuals.shape}")

    # 保存所有预测结果和实际值
    results_df = pd.DataFrame({
        '实际值': all_actuals,
        '预测值': all_predictions,
        '绝对误差': np.abs(all_predictions - all_actuals),
        '相对误差(%)': np.abs((all_predictions - all_actuals) / all_actuals * 100)
    })
    results_path = os.path.join(output_dir, f"{dataset_name}_cv_predictions.csv")
    results_df.to_csv(results_path, index=False, encoding='utf-8')

    # 保存每折的评估指标
    cv_df = pd.DataFrame(cv_metrics)
    # 将R²重命名为R^2以避免字体问题
    if 'R²' in cv_df.columns:
        cv_df.rename(columns={'R²': 'R^2'}, inplace=True)

    cv_df.index = [f"折{i + 1}" for i in range(len(cv_df))]
    cv_df.loc['平均值'] = cv_df.mean()
    cv_df.loc['标准差'] = cv_df.std()
    cv_path = os.path.join(output_dir, f"{dataset_name}_cv_metrics.csv")
    cv_df.to_csv(cv_path, encoding='utf-8')

    # 保存为JSON格式
    # 将R²重命名为R^2以避免字体问题
    if 'R²' in cv_metrics:
        cv_metrics['R^2'] = cv_metrics.pop('R²')

    save_experiment_results(
        {
            'predictions': all_predictions.tolist(),
            'actuals': all_actuals.tolist(),
            'cv_metrics': cv_metrics,
            'fold_metrics': fold_metrics
        },
        output_dir,
        f"{dataset_name}_cv_results"
    )

    print(f"交叉验证结果已保存到: {output_dir}")
    return results_df, cv_df


class ResultVisualizer:
    def __init__(self, dataset_name, output_dir):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def visualize_cv_results(self, all_predictions, all_actuals, cv_metrics=None):
        """可视化交叉验证结果"""
        # 可视化预测结果
        self.visualize_predictions(all_predictions, all_actuals)

        # 创建结果表格
        results_df, stats_df = self.create_results_table(all_predictions, all_actuals)

        # 如果有折叠指标，可视化每折结果
        if cv_metrics:
            # 将R²重命名为R^2以避免字体问题
            if 'R²' in cv_metrics:
                cv_metrics['R^2'] = cv_metrics.pop('R²')
            # 可视化每折的评估指标
            self.visualize_cv_metrics(cv_metrics)

    def visualize_predictions(self, predictions, actuals):
        """可视化预测结果"""
        # 确保输入是NumPy数组
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # 创建散点图 - 降低透明度到0.6
        plt.figure(figsize=(10, 6))
        plt.scatter(actuals, predictions, alpha=0.8, facecolor='none', edgecolor='#77AECD', s=30, linewidth=0.5)
        # plt.scatter(actuals, predictions, alpha=0.6, facecolor='none', edgecolor='#066190', s=30, linewidth=0.15)

        # 添加回归线 - 使用虚线，并限制显示范围在0.05到0.86之间
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(actuals, predictions)
        x = np.linspace(-0.02, 0.85, 100)  # 限制范围在0.05到0.86
        y = slope * x + intercept
        plt.plot(x, y, color='#3B86B9', linewidth=1, linestyle='--', alpha=0.8)  # 使用虚线并设置透明度

        # 计算评估指标
        r2 = r2_score(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        r = np.corrcoef(actuals, predictions)[0, 1]  # 计算相关系数r

        # 添加评估指标文本到左上角的方框中
        plt.text(0.05, 0.95,
                 f'r = {r:.4f}\n$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}',
                 transform=plt.gca().transAxes,
                 # bbox=dict(facecolor='white', edgecolor='#666666', boxstyle='round,pad=0.5'),
                 verticalalignment='top')

        # # 添加回归线信息到右下角
        # plt.text(0.95, 0.05,
        #          f'回归线: y={slope:.4f}x+{intercept:.4f}',
        #          transform=plt.gca().transAxes,
        #          bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3'),
        #          horizontalalignment='right',
        #          verticalalignment='bottom',
        #          color='blue')

        plt.title('PERSIANN-CDR')
        plt.xlabel('True In-situ-R')
        plt.ylabel('Predicted In-situ-R')

        # 设置坐标轴范围为-0.05到0.95
        plt.xlim(-0.05, 0.9)
        plt.ylim(-0.05, 0.9)

        # 设置显示刻度间隔为0.2
        plt.xticks(np.arange(0, 1.0, 0.2))
        plt.yticks(np.arange(0, 1.0, 0.2))

        # 设置网格线间隔为0.1
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.gca().set_xticks(np.arange(0, 1.0, 0.1), minor=True)
        plt.gca().set_yticks(np.arange(0, 1.0, 0.1), minor=True)
        plt.gca().xaxis.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
        plt.gca().yaxis.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
        plt.gca().xaxis.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.gca().yaxis.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)

        # 保存图片
        plt.savefig(os.path.join(self.output_dir, f'{self.dataset_name}_predictions.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 创建残差图
        plt.figure(figsize=(10, 6))
        residuals = predictions - actuals
        plt.scatter(actuals, residuals, alpha=0.6, color='#77AECD', edgecolor='#066190', s=30)
        plt.axhline(y=0, color='#666666', linestyle='--', linewidth=1.5)
        plt.title('残差分布图')
        plt.xlabel('实际值')
        plt.ylabel('残差(预测值-实际值)')

        # 设置x轴范围为-0.05到0.95
        plt.xlim(-0.1, 1.0)

        # 设置x轴显示刻度间隔为0.2
        plt.xticks(np.arange(0, 1.0, 0.2))

        # 设置网格线间隔为0.1
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.gca().set_xticks(np.arange(-0.05, 1.0, 0.1), minor=True)
        plt.gca().xaxis.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.3)
        plt.gca().xaxis.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.gca().yaxis.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)

        # 保存残差图
        plt.savefig(os.path.join(self.output_dir, f'{self.dataset_name}_residuals.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 创建残差直方图
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, alpha=0.7, color='#77AECD')
        plt.title('残差直方图')
        plt.xlabel('残差值')
        plt.ylabel('频率')

        # 添加网格线
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, axis='y')

        # 保存残差直方图
        plt.savefig(os.path.join(self.output_dir, f'{self.dataset_name}_residuals_hist.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

    def create_results_table(self, predictions, actuals):
        """创建结果对比表格"""
        # 确保输入是NumPy数组
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # 创建DataFrame
        results_df = pd.DataFrame({
            '实际值': actuals,
            '预测值': predictions,
            '误差': predictions - actuals,
            '相对误差(%)': ((predictions - actuals) / actuals * 100)
        })

        # 计算统计指标
        stats = {
            'r': np.corrcoef(actuals, predictions)[0, 1],  # 相关系数r
            'R^2': r2_score(actuals, predictions),  # 使用R^2代替R²
            'RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
            'MAE': mean_absolute_error(actuals, predictions),
            '平均相对误差(%)': np.mean(np.abs((predictions - actuals) / actuals * 100))
        }

        # 保存结果
        results_df.to_csv(os.path.join(self.output_dir, f'{self.dataset_name}_results.csv'), index=False)

        # 创建统计指标表格
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(os.path.join(self.output_dir, f'{self.dataset_name}_stats.csv'), index=False)

        return results_df, stats_df

    def visualize_cv_metrics(self, cv_metrics):
        """可视化交叉验证的评估指标"""
        metrics = list(cv_metrics.keys())
        folds = list(range(1, len(cv_metrics[metrics[0]]) + 1))

        # 为每个指标创建折线图
        plt.figure(figsize=(12, 8))

        for metric in metrics:
            plt.plot(folds, cv_metrics[metric], marker='o', label=metric)

        plt.title('交叉验证各折的评估指标')
        plt.xlabel('折叠')
        plt.ylabel('指标值')
        plt.xticks(folds)
        plt.legend()
        plt.grid(True)

        # 保存图片
        plt.savefig(os.path.join(self.output_dir, f'{self.dataset_name}_cv_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 创建柱状图比较
        plt.figure(figsize=(12, 8))
        bar_width = 0.2
        index = np.arange(len(folds))

        for i, metric in enumerate(metrics):
            plt.bar(index + i * bar_width, cv_metrics[metric], bar_width,
                    label=metric, alpha=0.7)

        plt.title('各折之间的评估指标比较')
        plt.xlabel('交叉验证折')
        plt.ylabel('指标值')
        plt.xticks(index + bar_width, [f'折{i}' for i in folds])
        plt.legend()
        plt.grid(True, axis='y')

        # 保存图片
        plt.savefig(os.path.join(self.output_dir, f'{self.dataset_name}_fold_comparison.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()


def collect_fold_results(input_dir, dataset_name, n_folds=5):
    """
    从多个独立的交叉验证折结果文件中收集数据

    参数:
    input_dir: 输入目录
    dataset_name: 数据集名称
    n_folds: 交叉验证折数

    返回:
    all_predictions: 合并后的所有预测值
    all_actuals: 合并后的所有实际值
    cv_metrics: 交叉验证指标
    fold_metrics: 每折的详细指标
    """
    all_predictions = []
    all_actuals = []
    cv_metrics = {'R^2': [], 'RMSE': [], 'MAE': []}  # 使用R^2代替R²
    fold_metrics = []

    # 遍历每个fold
    for fold_idx in range(1, n_folds + 1):
        # 尝试不同可能的文件名模式
        fold_patterns = [
            f"{dataset_name}_fold{fold_idx}_test.json",  # 模式1: dataset_fold1_test.json
            f"fold{fold_idx}/{dataset_name}_test.json",  # 模式2: fold1/dataset_test.json
            f"fold{fold_idx}_test.json",                # 模式3: fold1_test.json
            f"{dataset_name}_fold_{fold_idx}_test.json"  # 模式4: dataset_fold_1_test.json
        ]

        found = False
        for pattern in fold_patterns:
            fold_path = os.path.join(input_dir, pattern)
            if os.path.exists(fold_path):
                print(f"找到第{fold_idx}折结果文件: {fold_path}")
                try:
                    with open(fold_path, 'r', encoding='utf-8') as f:
                        fold_results = json.load(f)

                    # 提取预测值和实际值
                    if 'predictions' in fold_results and 'actuals' in fold_results:
                        fold_predictions = fold_results['predictions']
                        fold_actuals = fold_results['actuals']

                        # 添加到合并列表
                        all_predictions.extend(fold_predictions)
                        all_actuals.extend(fold_actuals)

                        # 提取指标
                        if 'metrics' in fold_results:
                            metrics = fold_results['metrics']

                            # 将R²重命名为R^2
                            if 'R²' in metrics:
                                metrics['R^2'] = metrics.pop('R²')

                            fold_metrics.append(metrics)

                            # 更新交叉验证指标
                            if 'R^2' in metrics:
                                cv_metrics['R^2'].append(metrics['R^2'])
                            if 'RMSE' in metrics:
                                cv_metrics['RMSE'].append(metrics['RMSE'])
                            if 'MAE' in metrics:
                                cv_metrics['MAE'].append(metrics['MAE'])

                        found = True
                        break
                except Exception as e:
                    print(f"读取第{fold_idx}折结果文件时出错: {e}")

        # 如果找不到任何模式的JSON结果文件，尝试CSV文件
        if not found:
            csv_patterns = [
                f"{dataset_name}_fold{fold_idx}_test.csv",  # 模式1: dataset_fold1_test.csv
                f"fold{fold_idx}/{dataset_name}_test.csv",  # 模式2: fold1/dataset_test.csv
                f"fold{fold_idx}_test.csv",                # 模式3: fold1_test.csv
                f"{dataset_name}_fold_{fold_idx}_test.csv"  # 模式4: dataset_fold_1_test.csv
            ]

            for pattern in csv_patterns:
                csv_path = os.path.join(input_dir, pattern)
                if os.path.exists(csv_path):
                    print(f"找到第{fold_idx}折CSV结果文件: {csv_path}")
                    try:
                        df = pd.read_csv(csv_path)
                        if '实际值' in df.columns and '预测值' in df.columns:
                            fold_predictions = df['预测值'].tolist()
                            fold_actuals = df['实际值'].tolist()

                            # 添加到合并列表
                            all_predictions.extend(fold_predictions)
                            all_actuals.extend(fold_actuals)

                            # 计算指标
                            r2 = r2_score(fold_actuals, fold_predictions)
                            rmse = np.sqrt(mean_squared_error(fold_actuals, fold_predictions))
                            mae = mean_absolute_error(fold_actuals, fold_predictions)

                            metrics = {'R^2': r2, 'RMSE': rmse, 'MAE': mae}  # 使用R^2代替R²
                            fold_metrics.append(metrics)

                            # 更新交叉验证指标
                            cv_metrics['R^2'].append(r2)
                            cv_metrics['RMSE'].append(rmse)
                            cv_metrics['MAE'].append(mae)

                            found = True
                            break
                        else:
                            print(f"CSV文件中找不到'实际值'或'预测值'列")
                    except Exception as e:
                        print(f"读取第{fold_idx}折CSV文件时出错: {e}")

        if not found:
            print(f"警告: 找不到第{fold_idx}折的结果文件")

    if not all_predictions or not all_actuals:
        print("未能从任何折中收集数据")
        return None, None, None, None

    print(f"已从{len(fold_metrics)}个折中收集数据")
    print(f"合并后共有{len(all_predictions)}个预测样本")

    return all_predictions, all_actuals, cv_metrics, fold_metrics


def main():
    """主函数"""
    # 配置参数
    # 可以从命令行参数接收，或者直接在脚本中设置
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        dataset_name = "cdr"  # 默认数据集名称

    if len(sys.argv) > 2:
        n_folds = int(sys.argv[2])
    else:
        n_folds = 5  # 默认5折交叉验证

    # 配置目录
    base_dir = "D:/hongyouting/result/dnn"
    input_dir = os.path.join(base_dir, dataset_name, "cv")
    output_dir = input_dir  # 输出到同一目录

    print(f"处理数据集: {dataset_name}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"交叉验证折数: {n_folds}")

    # 从各个折中收集数据
    all_predictions, all_actuals, cv_metrics, fold_metrics = collect_fold_results(input_dir, dataset_name, n_folds)

    if all_predictions is None or all_actuals is None:
        print("未能收集到交叉验证结果，程序退出。")
        return

    # 确认数据收集成功
    print(f"已收集预测结果: {len(all_predictions)} 个样本")
    print(f"已收集实际值: {len(all_actuals)} 个样本")
    print(f"已收集 {len(cv_metrics.get('R^2', []))} 折交叉验证结果")

    # 保存处理后的结果
    print("保存合并后的交叉验证结果...")
    save_cv_results(
        all_predictions, all_actuals,
        cv_metrics, fold_metrics,
        output_dir, dataset_name
    )

    # 创建可视化
    print("创建可视化图表...")
    visualizer = ResultVisualizer(dataset_name, output_dir)
    visualizer.visualize_cv_results(all_predictions, all_actuals, cv_metrics)

    print("处理完成！所有结果已保存到输出目录。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class ResultVisualizer:
    def __init__(self, dataset_name, output_dir):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def visualize_all(self, train_losses=None, val_losses=None, predictions=None, actuals=None):
        """可视化所有结果"""
        if train_losses is not None and val_losses is not None:
            self.visualize_losses(train_losses, val_losses)

        if predictions is not None and actuals is not None:
            self.visualize_predictions(predictions, actuals)
            self.create_results_table(predictions, actuals)

    def visualize_losses(self, train_losses, val_losses):
        """可视化训练和验证损失"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='训练损失', color='blue')
        plt.plot(val_losses, label='验证损失', color='red')
        plt.title('训练过程中的损失变化')
        plt.xlabel('轮次')
        plt.ylabel('损失值')
        plt.legend()
        plt.grid(True)

        # 保存图片
        plt.savefig(os.path.join(self.output_dir, f'{self.dataset_name}_losses.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_predictions(self, predictions, actuals):
        """可视化预测结果"""
        # 创建散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(actuals, predictions, alpha=0.5)

        # 添加对角线
        min_val = min(min(actuals), min(predictions))
        max_val = max(max(actuals), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想预测线')

        # 计算评估指标
        r2 = r2_score(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)

        # 添加评估指标文本
        plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}',
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

        plt.title('预测值 vs 实际值对比')
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.legend()
        plt.grid(True)

        # 保存图片
        plt.savefig(os.path.join(self.output_dir, f'{self.dataset_name}_predictions.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def create_results_table(self, predictions, actuals):
        """创建结果对比表格"""
        # 创建DataFrame
        results_df = pd.DataFrame({
            '实际值': actuals,
            '预测值': predictions,
            '误差': predictions - actuals,
            '相对误差(%)': ((predictions - actuals) / actuals * 100)
        })

        # 计算统计指标
        stats = {
            'R²': r2_score(actuals, predictions),
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
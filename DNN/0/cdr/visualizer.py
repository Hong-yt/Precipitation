import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
from config import Config
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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
        plt.plot(train_losses, label='训练损失(MSE)', color='blue')
        plt.plot(val_losses, label='验证损失(MSE)', color='red')
        plt.title('训练过程中的损失变化')
        plt.xlabel('轮次')
        plt.ylabel('MSE损失值')
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

        # 添加回归线
        slope, intercept, r_value, p_value, std_err = stats.linregress(actuals, predictions)
        x = np.linspace(min(actuals), max(actuals), 100)
        y = slope * x + intercept
        plt.plot(x, y, 'b-', label=f'回归线: y={slope:.4f}x+{intercept:.4f}')

        # 计算评估指标
        r2 = r2_score(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        r = np.corrcoef(actuals, predictions)[0, 1]  # 计算相关系数r

        # 添加评估指标文本，使用上标显示R²
        plt.text(0.05, 0.95, f'r = {r:.4f}\n$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}',
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

        plt.title('预测值 vs 实际值对比')
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.legend()
        plt.grid(True)

        # 保存图片
        plt.savefig(os.path.join(self.output_dir, f'{self.dataset_name}_predictions.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 创建残差图
        plt.figure(figsize=(10, 6))
        residuals = predictions - actuals
        plt.scatter(actuals, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('残差分布图')
        plt.xlabel('实际值')
        plt.ylabel('残差(预测值-实际值)')
        plt.grid(True)

        # 保存残差图
        plt.savefig(os.path.join(self.output_dir, f'{self.dataset_name}_residuals.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 创建残差直方图
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, alpha=0.7, color='blue')
        plt.title('残差直方图')
        plt.xlabel('残差值')
        plt.ylabel('频率')
        plt.grid(True)

        # 保存残差直方图
        plt.savefig(os.path.join(self.output_dir, f'{self.dataset_name}_residuals_hist.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

        # 调用双图表方法
        self.visualize_predictions_dual(predictions, actuals)

    def visualize_predictions_dual(self, predictions, actuals):
        """
        可视化原始和标准化的预测结果对比
        创建两张图：一张是原始数据，一张是标准化后的数据
        """
        # 计算评估指标
        r2 = r2_score(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        r = np.corrcoef(actuals, predictions)[0, 1]  # 计算相关系数r

        # 创建子图
        plt.figure(figsize=(18, 7))

        # 1. 绘制原始数据散点图
        plt.subplot(1, 2, 1)
        plt.scatter(actuals, predictions, alpha=0.5)

        # 添加回归线
        slope, intercept, r_value, p_value, std_err = stats.linregress(actuals, predictions)
        x = np.linspace(0, 1, 100)
        y = slope * x + intercept
        plt.plot(x, y, 'r--', label=f'回归线: y={slope:.4f}x+{intercept:.4f}')

        # 设置图表范围为0-1
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # 添加评估指标文本，确保R²正确显示
        plt.text(0.05, 0.95, f'r = {r:.4f}\n$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}',
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

        plt.title('原始数据预测值 vs 实际值')
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.legend()
        plt.grid(True)

        # 2. 绘制标准化数据散点图
        plt.subplot(1, 2, 2)

        # 标准化数据
        actuals_std = (actuals - np.mean(actuals)) / np.std(actuals)
        predictions_std = (predictions - np.mean(predictions)) / np.std(predictions)

        plt.scatter(actuals_std, predictions_std, alpha=0.5)

        # 添加标准化回归线
        slope_std, intercept_std, r_value_std, p_value_std, std_err_std = stats.linregress(actuals_std, predictions_std)
        x_std = np.linspace(min(actuals_std), max(actuals_std), 100)
        y_std = slope_std * x_std + intercept_std
        plt.plot(x_std, y_std, 'r--', label=f'回归线: y={slope_std:.4f}x+{intercept_std:.4f}')

        # 添加评估指标文本，确保R²正确显示
        plt.text(0.05, 0.95, f'r = {r:.4f}\n$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}',
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

        plt.title('标准化数据预测值 vs 实际值')
        plt.xlabel('标准化实际值')
        plt.ylabel('标准化预测值')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        # 保存图片
        plt.savefig(os.path.join(self.output_dir, f'{self.dataset_name}_predictions_dual.png'), dpi=300,
                    bbox_inches='tight')
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
            'r': np.corrcoef(actuals, predictions)[0, 1],  # 相关系数r
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

    def visualize_cv_results(self, all_predictions, all_actuals, cv_metrics=None):
        """可视化交叉验证结果"""
        # 可视化预测结果
        self.visualize_predictions(all_predictions, all_actuals)

        # 创建结果表格
        results_df, stats_df = self.create_results_table(all_predictions, all_actuals)

        # 如果有折叠指标，可视化每折结果
        if cv_metrics:
            # 可视化每折的评估指标
            self.visualize_cv_metrics(cv_metrics)

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

    def visualize_cv_loss_and_accuracy(self, fold_losses, fold_metrics):
        """
        可视化五折交叉验证的损失和精度

        参数:
        fold_losses: 每折的训练和验证损失，格式为 {fold_idx: {'train_losses': [...], 'val_losses': [...]}}
        fold_metrics: 每折的评估指标，格式为 list[dict]，每个dict包含R², RMSE, MAE等
        """
        # 创建一个子图布局，只显示最终结果
        plt.figure(figsize=(10, 6))

        # 绘制每一折的最终损失值
        fold_indices = list(fold_losses.keys())
        final_train_losses = [loss_data['train_losses'][-1] for fold_idx, loss_data in fold_losses.items()]
        final_val_losses = [loss_data['val_losses'][-1] for fold_idx, loss_data in fold_losses.items()]

        # 绘制条形图
        x = np.arange(len(fold_indices))
        width = 0.35

        plt.bar(x - width / 2, final_train_losses, width, label='训练损失')
        plt.bar(x + width / 2, final_val_losses, width, label='验证损失')

        plt.xlabel('交叉验证折')
        plt.ylabel('最终MSE损失')
        plt.title('各折最终损失值比较')
        plt.xticks(x, [f'第{i}折' for i in fold_indices])
        plt.legend()
        plt.grid(True, axis='y')

        # 保存图片
        plt.savefig(os.path.join(self.output_dir, f'{self.dataset_name}_cv_final_losses.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

        # 绘制每一折的最终R²值
        if fold_metrics and 'r2_history' in fold_metrics[0]:
            plt.figure(figsize=(10, 6))
            final_r2_values = [metrics['r2_history'][-1] if 'r2_history' in metrics else metrics.get('R²', 0)
                               for metrics in fold_metrics]

            plt.bar(x, final_r2_values, width=0.5)
            plt.xlabel('交叉验证折')
            plt.ylabel('$R^2$值')
            plt.title('各折最终$R^2$值比较')
            plt.xticks(x, [f'第{i + 1}折' for i in range(len(fold_metrics))])
            plt.grid(True, axis='y')

            # 保存图片
            plt.savefig(os.path.join(self.output_dir, f'{self.dataset_name}_cv_final_r2.png'), dpi=300,
                        bbox_inches='tight')
            plt.close()

    def visualize_fold_comparison(self, cv_metrics):
        """可视化不同折叠之间的性能比较"""
        # 准备数据
        metrics = list(cv_metrics.keys())
        num_folds = len(cv_metrics[metrics[0]])
        fold_names = [f'折{i + 1}' for i in range(num_folds)]

        # 创建柱状图
        plt.figure(figsize=(12, 8))
        bar_width = 0.2
        index = np.arange(num_folds)

        for i, metric in enumerate(metrics):
            plt.bar(index + i * bar_width, cv_metrics[metric], bar_width,
                    label=metric, alpha=0.7)

        plt.title('各折之间的评估指标比较')
        plt.xlabel('交叉验证折')
        plt.ylabel('指标值')
        plt.xticks(index + bar_width, fold_names)
        plt.legend()
        plt.grid(True, axis='y')

        # 保存图片
        plt.savefig(os.path.join(self.output_dir, f'{self.dataset_name}_fold_comparison.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

    def visualize_feature_importance(self, data_processor, fold_metrics=None):
        """
        可视化特征重要性

        参数:
        data_processor: 数据处理器对象，包含特征选择信息
        fold_metrics: 每折的评估指标，用于添加标题信息
        """
        # 检查是否有特征选择结果
        if not hasattr(data_processor, 'selected_features') or data_processor.selected_features is None:
            print("警告: 无特征重要性数据可用")
            return

        # 检查是否有特征名称
        if not hasattr(data_processor, 'feature_names') or data_processor.feature_names is None:
            feature_names = [f"特征{i}" for i in range(len(data_processor.selected_features))]
        else:
            # 获取所选特征的名称
            if len(data_processor.selected_features) < len(data_processor.feature_names):
                feature_names = [data_processor.feature_names[i] for i in data_processor.selected_features]
            else:
                feature_names = data_processor.feature_names

        # 创建特征重要性图，如果有随机森林模型的重要性分数就使用
        plt.figure(figsize=(12, 8))

        if hasattr(data_processor, 'feature_selector') and data_processor.feature_selector is not None:
            # 使用随机森林的特征重要性
            importances = data_processor.feature_selector.feature_importances_
            indices = np.argsort(importances)[::-1]  # 按重要性降序排序

            # 绘制前20个最重要的特征，或全部（如果少于20个）
            n_features = min(20, len(importances))
            plt.figure(figsize=(12, 8))
            plt.title('特征重要性排名')
            plt.bar(range(n_features), importances[indices[:n_features]], align='center', alpha=0.7)
            plt.xticks(range(n_features), [feature_names[i] for i in indices[:n_features]], rotation=45, ha='right')
            plt.tight_layout()
            plt.xlabel('特征')
            plt.ylabel('重要性分数')

        else:
            # 没有重要性分数，只显示选定的特征
            plt.title('选定的特征')
            plt.bar(range(len(feature_names)), [1] * len(feature_names), align='center', alpha=0.7)
            plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
            plt.tight_layout()
            plt.xlabel('特征')
            plt.ylabel('已选择')

        # 如果有交叉验证指标，添加到标题
        if fold_metrics:
            mean_r2 = np.mean([m.get('R²', 0) for m in fold_metrics])
            plt.title(f'特征重要性 (平均 R² = {mean_r2:.4f})')

        # 保存图片
        plt.savefig(os.path.join(self.output_dir, f'{self.dataset_name}_feature_importance.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

        # 创建特征分布的箱线图（如果原始数据可用）
        if hasattr(data_processor, 'features_df') and data_processor.features_df is not None:
            # 限制为前10个特征，避免图表过于复杂
            n_box_features = min(10, len(feature_names))
            plt.figure(figsize=(14, 8))
            data_processor.features_df.iloc[:, :n_box_features].boxplot()
            plt.title('前10个特征的分布')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{self.dataset_name}_feature_distributions.png'), dpi=300,
                        bbox_inches='tight')
            plt.close()

            # 创建相关性热图
            plt.figure(figsize=(14, 12))
            corr_matrix = data_processor.features_df.corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False,
                        vmax=1, vmin=-1, center=0, square=True, linewidths=.5)
            plt.title('特征相关性热图')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{self.dataset_name}_correlation_heatmap.png'), dpi=300,
                        bbox_inches='tight')
            plt.close()
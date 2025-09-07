import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
from config import Config
from scipy import stats
import shap
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 忽略matplotlib的警告
warnings.filterwarnings("ignore", category=UserWarning)

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
        # 确保输入是numpy数组
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        if isinstance(actuals, list):
            actuals = np.array(actuals)

        # 创建散点图
        plt.figure(figsize=(10, 6))
        # plt.scatter(actuals, predictions, alpha=0.5)
        plt.scatter(actuals, predictions, alpha=0.8, facecolor='none', edgecolor='#77AECD', s=30, linewidth=0.5)

        # # 添加回归线
        # slope, intercept, r_value, p_value, std_err = stats.linregress(actuals, predictions)
        # x = np.linspace(min(actuals), max(actuals), 100)
        # y = slope * x + intercept
        # plt.plot(x, y, 'b-', label=f'回归线: y={slope:.4f}x+{intercept:.4f}')

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

        # # 添加评估指标文本，使用上标显示R²
        # plt.text(0.05, 0.95, f'r = {r:.4f}\n$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}',
        #          transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))

        # 添加评估指标文本到左上角的方框中
        plt.text(0.05, 0.95,
                 f'r = {r:.4f}\n$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}',
                 transform=plt.gca().transAxes,
                 # bbox=dict(facecolor='white', edgecolor='#666666', boxstyle='round,pad=0.5'),
                 verticalalignment='top')

        plt.title('GSMaP_nrtg')
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
        # 确保输入是numpy数组
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        if isinstance(actuals, list):
            actuals = np.array(actuals)

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
        # 确保输入是numpy数组
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        if isinstance(actuals, list):
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
        data_processor: 数据处理器，包含特征名称和特征数据
        fold_metrics: 各折的评估指标，用于添加模型性能信息
        """
        # 如果没有特征名称，就不进行可视化
        if data_processor is None or not hasattr(data_processor, 'get_feature_names'):
            print("警告: 数据处理器缺少get_feature_names方法，无法可视化特征重要性")
            return

        feature_names = data_processor.get_feature_names()
        if not feature_names:
            print("警告: 无有效特征名称，无法可视化特征重要性")
            return

        # 显示特征分布和相关性
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

    def visualize_shap_feature_importance(self, importance_df, top_n=23):
        """
        可视化SHAP特征重要性

        参数:
        importance_df: 包含特征重要性的DataFrame或文件路径
        top_n: 显示前N个最重要的特征
        """
        try:
            # 如果输入是字符串，假设它是文件路径
            if isinstance(importance_df, str) and os.path.exists(importance_df):
                importance_df = pd.read_csv(importance_df)

            # 如果是字典列表，转换为DataFrame
            if isinstance(importance_df, list) and all(isinstance(item, dict) for item in importance_df):
                importance_df = pd.DataFrame(importance_df)

            # 检查输入是否为DataFrame
            if not isinstance(importance_df, pd.DataFrame):
                print(f"错误: importance_df必须是pandas DataFrame，而不是 {type(importance_df)}")
                return

            # 确保DataFrame包含必要的列
            if 'Feature' not in importance_df.columns or 'Importance' not in importance_df.columns:
                # 尝试推断列名
                if len(importance_df.columns) >= 2:
                    # 假设第一列是特征名称，第二列是重要性值
                    importance_df.columns = ['Feature', 'Importance'] + list(importance_df.columns[2:])
                    print("警告: 已自动推断列名为'Feature'和'Importance'")
                else:
                    print("错误: importance_df必须包含'Feature'和'Importance'列")
                    return

            # 确保重要性值是数值型
            importance_df['Importance'] = pd.to_numeric(importance_df['Importance'], errors='coerce')

            # 移除任何包含NaN的行
            importance_df = importance_df.dropna(subset=['Importance'])

            if len(importance_df) == 0:
                print("错误: 处理后的importance_df为空")
                return

            # 按重要性降序排序
            importance_df = importance_df.sort_values('Importance', ascending=False)

            # 取前N个特征
            if top_n > 0 and len(importance_df) > top_n:
                importance_df = importance_df.head(top_n)

            # 创建水平条形图
            plt.figure(figsize=(12, max(8, len(importance_df) * 0.4)))

            # 创建条形图
            bars = plt.barh(importance_df['Feature'], importance_df['Importance'],
                            color='#1f77b4', alpha=0.8, edgecolor='grey')

            # 添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{width:.4f}', va='center', fontsize=9)

            # 设置图表属性
            plt.xlabel('平均|SHAP值|（特征重要性）')
            plt.title(f'SHAP特征重要性（前{len(importance_df)}个特征）')
            plt.gca().invert_yaxis()  # 使最重要的特征显示在顶部
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()

            # 保存图片
            output_path = os.path.join(self.output_dir, f'{self.dataset_name}_shap_importance.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"特征重要性图表已保存到 {output_path}")

            # 如果重要性DataFrame包含折叠信息，创建按折叠分组的图表
            if 'Fold' in importance_df.columns:
                # 按折叠分组并创建多系列条形图
                try:
                    fold_groups = importance_df.groupby('Fold')
                    fold_indices = sorted(fold_groups.groups.keys())
                    n_folds = len(fold_indices)

                    if n_folds > 1:
                        # 获取所有特征并合并
                        all_features = set()
                        for fold_idx, group in fold_groups:
                            top_features = group.sort_values('Importance', ascending=False).head(min(top_n, len(group)))
                            all_features.update(top_features['Feature'])

                        # 限制特征数量
                        if len(all_features) > top_n * 2:
                            # 合并所有折的数据，按平均重要性选择特征
                            avg_importance = importance_df.groupby('Feature')['Importance'].mean().reset_index()
                            avg_importance = avg_importance.sort_values('Importance', ascending=False)
                            selected_features = set(avg_importance.head(top_n)['Feature'])
                        else:
                            selected_features = all_features

                        # 创建一个新的DataFrame，包含所有折的选定特征
                        fold_importance = pd.DataFrame()
                        for fold_idx, group in fold_groups:
                            # 选择特定折中的选定特征
                            fold_data = group[group['Feature'].isin(selected_features)]
                            fold_data = fold_data.sort_values('Importance', ascending=False)

                            # 添加到合并的DataFrame
                            fold_importance = pd.concat([fold_importance, fold_data])

                        # 创建多系列条形图
                        plt.figure(figsize=(14, max(10, len(selected_features) * 0.4)))

                        # 按特征透视数据
                        pivot_df = fold_importance.pivot(index='Feature', columns='Fold', values='Importance')

                        # 按平均重要性排序
                        pivot_df['Mean'] = pivot_df.mean(axis=1)
                        pivot_df = pivot_df.sort_values('Mean', ascending=True)
                        pivot_df = pivot_df.drop('Mean', axis=1)

                        # 绘制条形图
                        pivot_df.plot(kind='barh', figsize=(14, max(10, len(pivot_df) * 0.5)),
                                      alpha=0.7, edgecolor='grey', ax=plt.gca())

                        # 添加平均值线
                        avg_values = []
                        for feature in pivot_df.index:
                            # 计算此特征在所有折的平均值
                            vals = pivot_df.loc[feature].dropna()
                            if len(vals) > 0:
                                avg = vals.mean()
                                avg_values.append((feature, avg))

                        # 绘制平均值线
                        for feature, avg in avg_values:
                            y_pos = pivot_df.index.get_loc(feature)
                            plt.plot([0, avg], [y_pos, y_pos], 'k--', alpha=0.3)
                            plt.plot([avg, avg], [y_pos - 0.4, y_pos + 0.4], 'k-', linewidth=2)
                            plt.text(avg + 0.01, y_pos, f'平均: {avg:.4f}', va='center', fontsize=8)

                        # 设置图表属性
                        plt.xlabel('平均|SHAP值|（特征重要性）')
                        plt.title(f'各折叠的SHAP特征重要性（前{len(selected_features)}个特征）')
                        plt.grid(axis='x', linestyle='--', alpha=0.6)
                        plt.legend(title='折叠')
                        plt.tight_layout()

                        # 保存图片
                        output_path = os.path.join(self.output_dir, f'{self.dataset_name}_shap_importance_by_fold.png')
                        plt.savefig(output_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"按折叠分组的特征重要性图表已保存到 {output_path}")

                except Exception as e:
                    print(f"创建按折叠分组的特征重要性图表时出错: {e}")

        except Exception as e:
            print(f"可视化SHAP特征重要性时出错: {e}")
            import traceback
            traceback.print_exc()

    def visualize_shap_summary(self, shap_values, features, feature_names=None, max_display=23, plot_type="bar"):
        """
        可视化SHAP摘要图

        参数:
        shap_values: SHAP值矩阵
        features: 特征数据
        feature_names: 特征名称列表
        max_display: 最多显示的特征数量
        plot_type: 图表类型，可选 "bar", "dot", "violin"
        """
        try:
            # 检查数据有效性
            if shap_values is None or features is None or len(shap_values) == 0 or len(features) == 0:
                print("错误: 无效的SHAP值或特征数据")
                return

            # 替换NaN值，以防止可视化错误
            shap_values_clean = np.nan_to_num(shap_values)
            features_clean = np.nan_to_num(features)

            # 检查形状是否一致
            if shap_values_clean.shape[0] != features_clean.shape[0]:
                print(f"错误: SHAP值和特征数据形状不一致 {shap_values_clean.shape} vs {features_clean.shape}")
                return

            # 确保有特征名称
            if feature_names is None:
                feature_names = [f"特征_{i}" for i in range(features_clean.shape[1])]

            # 确保特征名称数量与特征数量一致
            if len(feature_names) != features_clean.shape[1]:
                print(f"警告: 特征名称数量 ({len(feature_names)}) 与特征数量 ({features_clean.shape[1]}) 不匹配")
                feature_names = [f"特征_{i}" for i in range(features_clean.shape[1])]

            # 计算特征重要性并排序
            feature_importance = np.abs(shap_values_clean).mean(0)
            sorted_idx = np.argsort(-feature_importance)
            top_idx = sorted_idx[:max_display]

            plt.figure(figsize=(10, 12))

            try:
                # 使用低级API替代shap的高级可视化函数
                if plot_type == "bar":
                    # 创建条形图
                    mean_abs_shap = np.abs(shap_values_clean).mean(0)
                    plt.barh(range(len(top_idx)), mean_abs_shap[top_idx])
                    plt.yticks(range(len(top_idx)), [feature_names[i] for i in top_idx])
                    plt.gca().invert_yaxis()  # 最大值在顶部
                    plt.xlabel('平均|SHAP值|')

                elif plot_type == "dot":
                    # 创建散点图/蜂群图
                    plt.figure(figsize=(12, max(8, 0.3 * len(top_idx))))

                    # 按特征重要性排序并绘制前max_display个特征
                    for i, idx in enumerate(top_idx):
                        # 获取特征值和SHAP值
                        feature_values = features_clean[:, idx]
                        shap_values_feature = shap_values_clean[:, idx]

                        # 按特征值排序
                        sorted_indices = np.argsort(feature_values)

                        # 绘制散点图
                        plt.scatter(
                            shap_values_feature[sorted_indices],
                            [i] * len(shap_values_feature),
                            c=feature_values[sorted_indices],
                            cmap='viridis',
                            alpha=0.7,
                            s=10
                        )

                    plt.yticks(range(len(top_idx)), [feature_names[i] for i in top_idx])
                    plt.xlabel('SHAP值')
                    plt.colorbar(label='特征值')
                    plt.grid(True, axis='x')

                elif plot_type == "violin":
                    # 为了可靠性，创建箱线图而不是小提琴图
                    plt.figure(figsize=(12, max(8, 0.3 * len(top_idx))))

                    # 准备数据
                    boxplot_data = []
                    for idx in top_idx:
                        boxplot_data.append(shap_values_clean[:, idx])

                    # 绘制箱线图
                    plt.boxplot(
                        boxplot_data,
                        vert=False,  # 水平方向的箱线图
                        labels=[feature_names[i] for i in top_idx]
                    )
                    plt.xlabel('SHAP值')
                    plt.grid(True, axis='x')

                plt.title(f"SHAP值摘要图 - {plot_type.capitalize()}图")

            except Exception as e:
                print(f"绘制SHAP摘要图出错 ({plot_type}): {e}")
                import traceback
                traceback.print_exc()
                plt.text(0.5, 0.5, f"摘要图生成失败 ({plot_type})", ha='center', va='center', fontsize=14)
                plt.title(f"SHAP值摘要图 - {plot_type.capitalize()}图 - 生成失败")

            plt.tight_layout()

            # 保存图表
            fold_str = ""
            if "_fold" in self.dataset_name:
                fold_str = self.dataset_name.split("_fold")[1]
                if fold_str.isdigit():
                    fold_str = f"_fold{fold_str}"
                else:
                    fold_str = ""

            plt.savefig(os.path.join(self.output_dir, f"{self.dataset_name}_shap_summary_{plot_type}.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"可视化SHAP摘要图出错: {e}")
            import traceback
            traceback.print_exc()

    def visualize_shap_feature_groups(self, shap_values, features, feature_names=None):
        """
        创建特征重要性和SHAP依赖图的组合可视化

        参数:
        shap_values: SHAP值矩阵
        features: 特征数据（原始未标准化的特征值）
        feature_names: 特征名称列表
        """
        try:
            # 检查数据有效性
            if shap_values is None or features is None or len(shap_values) == 0 or len(features) == 0:
                print("错误: 无效的SHAP值或特征数据")
                return

            # 替换NaN值，以防止可视化错误
            shap_values_clean = np.nan_to_num(shap_values)
            features_clean = np.nan_to_num(features)

            # 确保有特征名称
            if feature_names is None:
                feature_names = [f"特征_{i}" for i in range(features_clean.shape[1])]

            # 创建一个功能更丰富的全局SHAP解释图
            plt.figure(figsize=(10, 12))

            # 计算特征重要性（平均绝对SHAP值）
            importance = np.abs(shap_values_clean).mean(0)

            # 特征排序
            feature_order = np.argsort(importance)[::-1]

            # 使用所有特征（不限制特征数量）
            feature_names_ordered = [feature_names[i] for i in feature_order]

            # 特征重要性部分
            plt.title(f'特征重要性 - {self.dataset_name}', fontsize=14)

            # 显示特征重要性条形图
            plt.barh(range(len(feature_names_ordered)), importance[feature_order])
            plt.yticks(range(len(feature_names_ordered)), feature_names_ordered)
            plt.xlabel('平均|SHAP值|')  # 添加X轴标签
            plt.gca().invert_yaxis()  # 最大值在顶部

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{self.dataset_name}_feature_importance_all.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()

            # 创建SHAP依赖图 (类似参考图)
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'SHAP依赖图 (前6大特征) - {self.dataset_name}', fontsize=16)
            axes = axes.flatten()

            # 获取前6个最重要的特征
            top_n = min(6, len(feature_order))
            top_features_idx = feature_order[:top_n]

            # 尝试从数据集中估计每个特征的原始分布范围
            # 对于多数特征，标准化后的区间约为[-3, 3]，对应原始数据的[μ-3σ, μ+3σ]
            feature_ranges = {}
            for idx in top_features_idx:
                feature_data = features_clean[:, idx]
                feature_min, feature_max = np.min(feature_data), np.max(feature_data)

                # 假设以下是可能的原始范围，基于特征类型和名称来估计
                if "R" in feature_names[idx] or "r" in feature_names[idx].lower():
                    # 相关系数类型的特征，范围应该在[0, 1]
                    feature_ranges[idx] = (0, 1)
                else:
                    # 其他特征，默认范围保持其数据范围
                    feature_ranges[idx] = (feature_min, feature_max)

            # 创建2x3网格的子图
            for i, idx in enumerate(top_features_idx):
                ax = axes[i]

                # 获取特定特征的数据
                x = features_clean[:, idx]
                y = shap_values_clean[:, idx]

                # 确定特征范围
                if idx in feature_ranges:
                    feature_range = feature_ranges[idx]
                else:
                    # 默认范围，尝试使用一些启发式方法估计
                    feature_min, feature_max = np.min(x), np.max(x)
                    range_width = feature_max - feature_min
                    feature_range = (feature_min - 0.1 * range_width, feature_max + 0.1 * range_width)

                # 设置x轴范围
                if "R" in feature_names[idx] or "r" in feature_names[idx].lower():
                    # 对于相关系数，使用0-1范围
                    ax.set_xlim(0, 1)
                else:
                    # 其他特征使用数据范围
                    x_5th = np.percentile(x, 5)
                    x_95th = np.percentile(x, 95)
                    margin = (x_95th - x_5th) * 0.1  # 增加10%的边距
                    ax.set_xlim(x_5th - margin, x_95th + margin)

                # 绘制散点图
                scatter = ax.scatter(x, y, alpha=0.6, c=y, cmap='coolwarm', s=20)

                # 添加局部平滑趋势线
                try:
                    from scipy.stats import gaussian_kde
                    from scipy.ndimage import gaussian_filter1d

                    # 对散点数据排序，以便绘制线
                    sorted_idx = np.argsort(x)
                    x_sorted = x[sorted_idx]
                    y_sorted = y[sorted_idx]

                    # 使用高斯平滑
                    y_smooth = gaussian_filter1d(y_sorted, sigma=len(y_sorted) // 50)

                    # 绘制趋势线 - 红色
                    ax.plot(x_sorted, y_smooth, color='#d16d5b', linestyle='--', linewidth=1, alpha=0.8)
                except Exception as e:
                    print(f"绘制趋势线时出错: {e}")

                # 添加水平线表示SHAP值为0
                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

                # 设置Y轴范围，使正负值对称
                y_max = max(abs(np.min(y)), abs(np.max(y))) * 1.1
                ax.set_ylim(-y_max, y_max)

                # 添加栅格
                ax.grid(True, linestyle='--', alpha=0.3)

                # 只在底部显示特征名称
                ax.set_xlabel(feature_names[idx], fontsize=10)

                # 只在左侧两个图（索引0和3）显示Y轴标签
                if i == 0 or i == 3:
                    ax.set_ylabel('SHAP Value', fontsize=10)
                else:
                    ax.set_ylabel('')
                #
                # # 创建twin x轴用于显示In-situ-R在顶部
                # ax_twin = ax.twinx()
                #
                # # 在右侧轴显示In-situ-R值，颜色设为红色
                # ax_twin.set_ylim(0, 1)  # 相关系数的范围是0-1
                #
                # # 只在最右侧两个图显示In-situ-R标签，颜色设为红色
                # if i == 2 or i == 5:  # 第三列的两个图
                #     ax_twin.set_ylabel('In-situ-R', color='red')
                # else:
                #     ax_twin.set_ylabel('')
                #
                # # 设置In-situ-R的范围，确保下面是0上面是1
                # ax_twin.set_ylim(0, 1)
                # ax_twin.tick_params(axis='y', labelcolor='red')  # 设置刻度标签为红色

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为整体标题留出空间
            plt.savefig(os.path.join(self.output_dir, f"{self.dataset_name}_shap_dependence_top6.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()

            print(f"SHAP依赖图已保存至 {self.output_dir}/{self.dataset_name}_shap_dependence_top6.png")

        except Exception as e:
            print(f"创建SHAP分析图出错: {e}")
            import traceback
            traceback.print_exc()

    def visualize_shap_beeswarm(self, shap_values, features, feature_names=None, max_display=30):
        """
        创建SHAP值的蜂群图(beeswarm plot)，展示特征如何影响模型输出

        参数:
        shap_values: SHAP值矩阵
        features: 特征数据（标准化后的特征值效果更好）
        feature_names: 特征名称列表
        max_display: 最多显示的特征数量
        """
        try:
            # 检查数据有效性
            if shap_values is None or features is None or len(shap_values) == 0 or len(features) == 0:
                print("错误: 无效的SHAP值或特征数据")
                return

            # 替换NaN值，以防止可视化错误
            shap_values_clean = np.nan_to_num(shap_values)
            features_clean = np.nan_to_num(features)

            # 确保有特征名称
            if feature_names is None:
                feature_names = [f"特征_{i}" for i in range(features_clean.shape[1])]

            # 计算特征重要性并排序
            feature_importance = np.abs(shap_values_clean).mean(0)
            sorted_idx = np.argsort(-feature_importance)

            # 限制显示的特征数量
            if max_display > len(sorted_idx):
                max_display = len(sorted_idx)

            # 选择最重要的特征索引和名称
            top_idx = sorted_idx[:max_display]
            top_names = [feature_names[i] for i in top_idx]

            # 创建图表，设置白色背景
            plt.figure(figsize=(10, 12))
            plt.subplots_adjust(left=0.4)  # 为长特征名称留出更多空间

            # 使用更接近示例的颜色映射
            cmap = plt.cm.viridis_r  # 从深色到浅色的渐变

            # 添加小标题
            plt.suptitle('(b) Feature importance', fontsize=14, y=0.98)

            # 为每个特征创建散点图
            for i, idx in enumerate(top_idx):
                # 获取当前特征的SHAP值和特征值
                feature_shap_values = shap_values_clean[:, idx]
                feature_values = features_clean[:, idx]

                # 过滤掉NaN或inf值
                valid_indices = np.isfinite(feature_shap_values) & np.isfinite(feature_values)
                feature_shap_values = feature_shap_values[valid_indices]
                feature_values = feature_values[valid_indices]

                if len(feature_values) == 0:
                    continue

                # 归一化特征值以用于颜色映射
                if len(np.unique(feature_values)) > 1:
                    norm_values = (feature_values - np.min(feature_values)) / (
                            np.max(feature_values) - np.min(feature_values))
                else:
                    norm_values = np.zeros_like(feature_values)

                # 使用较小的点大小和更均匀的分布
                y_pos = max_display - 1 - i

                # 基于SHAP值的分布添加适当的抖动，使点分散开
                # 计算每个SHAP值的水平位置
                x_jitter = np.random.normal(scale=0.01, size=len(feature_shap_values))

                # 计算每个点的垂直抖动，使点分布更像示例图中的密度展示
                y_jitter = np.random.normal(scale=0.1, size=len(feature_shap_values))

                # 所有点使用相同大小
                point_size = 12

                # 绘制散点图
                plt.scatter(
                    feature_shap_values + x_jitter,  # x轴：SHAP值（加轻微水平抖动）
                    np.ones(len(feature_shap_values)) * y_pos + y_jitter,  # y轴：特征位置（加垂直抖动）
                    c=feature_values,  # 颜色：特征值
                    cmap=cmap,  # 颜色映射：从深色到浅色的渐变
                    s=point_size,  # 固定点大小
                    alpha=0.7,  # 稍高的透明度
                    edgecolor=None,  # 无边框
                    vmin=-2,  # 设置颜色标度的最小值
                    vmax=2  # 设置颜色标度的最大值
                )

            # 设置Y轴刻度和标签
            plt.yticks(range(max_display), top_names)

            # 添加坐标轴标签和标记
            plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)  # 0线
            plt.xlabel('SHAP value', fontsize=12)

            # 设置X轴范围，使图表更接近示例
            max_shap = np.max(np.abs(shap_values_clean))
            plt.xlim(-max_shap * 1.1, max_shap * 1.1)

            # 添加垂直网格线
            plt.grid(True, axis='x', linestyle='--', alpha=0.3)

            # 添加颜色条在右侧
            cbar = plt.colorbar(pad=0.01, aspect=40)
            cbar.set_label('Feature value', fontsize=10)

            # 增加坐标轴刻度的可见性
            plt.tick_params(axis='both', which='major', labelsize=10)

            # 保存图表
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{self.dataset_name}_shap_beeswarm.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()

            print(f"SHAP蜂群图已保存至 {self.output_dir}/{self.dataset_name}_shap_beeswarm.png")

        except Exception as e:
            print(f"创建SHAP蜂群图时出错: {e}")
            import traceback
            traceback.print_exc()

    def visualize_shap_combined(self, shap_values, features, feature_names=None, max_display=None):
        """
        创建组合图：左侧为平均SHAP值特征重要性的条形图，右侧为SHAP值的蜂群分布图

        参数:
        shap_values: SHAP值矩阵
        features: 特征数据（标准化后的特征值效果更好）
        feature_names: 特征名称列表
        max_display: 最多显示的特征数量，设为None则显示所有特征
        """
        try:
            # 检查数据有效性
            if shap_values is None or features is None or len(shap_values) == 0 or len(features) == 0:
                print("错误: 无效的SHAP值或特征数据")
                return

            # 替换NaN值，以防止可视化错误
            shap_values_clean = np.nan_to_num(shap_values)
            features_clean = np.nan_to_num(features)

            # 确保有特征名称
            if feature_names is None:
                feature_names = [f"特征_{i}" for i in range(features_clean.shape[1])]

            # 计算特征重要性并排序
            feature_importance = np.abs(shap_values_clean).mean(0)
            sorted_idx = np.argsort(-feature_importance)

            # 显示所有特征（不限制数量）
            if max_display is None or max_display > len(sorted_idx):
                max_display = len(sorted_idx)

            # 选择最重要的特征索引和名称
            top_idx = sorted_idx[:max_display]
            top_names = [feature_names[i] for i in top_idx]

            # 创建带有两个子图的组合图表，减少空白区域
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10), gridspec_kw={'width_ratios': [1, 2]})

            # ===== 左侧：平均SHAP值条形图 =====
            # 计算特征重要性
            mean_abs_shap = np.abs(shap_values_clean).mean(0)
            top_mean_abs_shap = mean_abs_shap[top_idx]

            # 为条形图创建浅蓝色背景
            for i in range(max_display):
                ax1.axhspan(i - 0.4, i + 0.4, color='#E6F3FF', alpha=0.6)

            # 绘制水平条形图
            bars = ax1.barh(range(max_display), top_mean_abs_shap, color='#4292C6', height=0.7, alpha=0.8)

            # 设置Y轴刻度（特征名称）和标签
            ax1.set_yticks(range(max_display))
            ax1.set_yticklabels(top_names)
            ax1.invert_yaxis()  # 最大值在顶部

            # 设置网格线
            ax1.grid(axis='x', linestyle='--', alpha=0.3)

            # 设置X轴范围从0开始
            ax1.set_xlim(0, max(top_mean_abs_shap) * 1.1)

            # 添加标题
            ax1.set_title('Mean Shapley Value (Feature Importance)', fontsize=12, pad=10)

            # ===== 右侧：蜂群图 =====
            # 使用更清晰的颜色映射
            cmap = plt.cm.coolwarm

            # 为所有特征创建蜂群图
            for i, idx in enumerate(top_idx):
                # 获取当前特征的SHAP值和特征值
                feature_shap_values = shap_values_clean[:, idx]
                feature_values = features_clean[:, idx]

                # 过滤掉NaN或inf值
                valid_indices = np.isfinite(feature_shap_values) & np.isfinite(feature_values)
                feature_shap_values = feature_shap_values[valid_indices]
                feature_values = feature_values[valid_indices]

                if len(feature_values) == 0:
                    continue

                # 为蜂群图创建浅蓝色背景
                ax2.axhspan(i - 0.4, i + 0.4, color='#E6F3FF', alpha=0.6)

                # 计算每个点的垂直抖动，使点分布更像密度
                y_jitter = np.random.normal(scale=0.1, size=len(feature_shap_values))

                # 绘制散点图，优化点大小和透明度
                sc = ax2.scatter(
                    feature_shap_values,  # x轴：SHAP值
                    np.ones(len(feature_shap_values)) * i + y_jitter,  # y轴：特征位置（加垂直抖动）
                    c=feature_values,  # 颜色：特征值
                    cmap=cmap,  # 颜色映射
                    s=25,  # 点大小稍微增加
                    alpha=0.75,  # 透明度稍微降低
                    edgecolor='none'  # 无边框
                )

            # 添加垂直线表示SHAP值为0
            ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.5)

            # 设置Y轴刻度和标签
            ax2.set_yticks(range(max_display))
            ax2.set_yticklabels([''] * max_display)  # 空标签，因为左侧已经有了
            ax2.invert_yaxis()  # 最大值在顶部

            # 设置网格线
            ax2.grid(axis='x', linestyle='--', alpha=0.3)

            # 设置X轴标签
            ax2.set_title('Shapley Value Contribution', fontsize=12, pad=10)


            # 计算X轴范围使其更紧凑
            max_shap = np.max(np.abs(shap_values_clean))
            ax2.set_xlim(-max_shap * 1.05, max_shap * 1.05)

            # 添加颜色条
            cbar = plt.colorbar(sc, ax=ax2, pad=0.02, aspect=30)

            # 移除颜色条刻度并添加High和Low标签
            cbar.set_ticks([])

            # 在颜色条最低处添加"Low"标签
            cbar.ax.text(0.5, -0.01, 'Low', ha='center', va='top', fontsize=10,
                         transform=cbar.ax.transAxes)

            # 在颜色条最高处添加"High"标签
            cbar.ax.text(0.5, 1.01, 'High', ha='center', va='bottom', fontsize=10,
                         transform=cbar.ax.transAxes)

            # 添加特征值标签
            cbar.set_label('Feature Value', fontsize=12, rotation=270, labelpad=15)

            # 设置整体标题
            plt.suptitle(f'SHAP Summary: {self.dataset_name}', fontsize=16, y=0.98)

            # 优化布局，减少空白区域
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(self.output_dir, f"{self.dataset_name}_shap_combined.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()

            print(f"SHAP组合图已保存至 {self.output_dir}/{self.dataset_name}_shap_combined.png")

        except Exception as e:
            print(f"创建SHAP组合图时出错: {e}")
            import traceback
            traceback.print_exc()

    def visualize_shap_force_plot(self, shap_values, features, feature_names=None, sample_indices=None, num_samples=5,
                                  output_dir=None):
        """
        可视化SHAP力图(force plot)，展示特定样本的特征贡献

        参数:
        shap_values: SHAP值矩阵
        features: 特征数据（原始未标准化的特征值）
        feature_names: 特征名称列表
        sample_indices: 要展示的样本索引列表，如果为None则随机选择
        num_samples: 如果sample_indices为None，要随机选择的样本数量
        output_dir: 输出目录，如果为None则使用类初始化时的目录
        """
        # 数据验证 - 更全面的错误处理
        if shap_values is None or features is None:
            print("错误: SHAP值或特征数据为空")
            return

        if isinstance(shap_values, list) and len(shap_values) == 0:
            print("错误: SHAP值列表为空")
            return

        if isinstance(features, list) and len(features) == 0:
            print("错误: 特征数据列表为空")
            return

        # 统一数据格式为numpy数组
        try:
            if not isinstance(shap_values, np.ndarray):
                shap_values = np.array(shap_values)
            if not isinstance(features, np.ndarray):
                features = np.array(features)
        except Exception as e:
            print(f"错误: 转换数据为numpy数组时失败: {e}")
            return

        # 替换NaN和Inf值，以防止可视化错误
        shap_values_clean = np.nan_to_num(shap_values)
        features_clean = np.nan_to_num(features)

        # 检查形状是否匹配
        if shap_values_clean.shape[0] != features_clean.shape[0]:
            print(
                f"错误: SHAP值和特征数据样本数量不匹配 - SHAP: {shap_values_clean.shape[0]}, 特征: {features_clean.shape[0]}")
            return

        # 确保有特征名称并且数量正确
        if feature_names is None:
            feature_names = [f"特征_{i}" for i in range(features_clean.shape[1])]
        elif len(feature_names) != features_clean.shape[1]:
            print(
                f"警告: 特征名称数量({len(feature_names)})与特征数量({features_clean.shape[1]})不匹配，将使用默认特征名称")
            feature_names = [f"特征_{i}" for i in range(features_clean.shape[1])]

        # 生成要展示的样本索引 - 增强的样本选择
        num_available = shap_values_clean.shape[0]
        if num_available == 0:
            print("错误: 没有可用的样本进行可视化")
            return

        if sample_indices is None:
            # 随机选择样本
            if num_available < num_samples:
                print(f"警告: 可用样本数量({num_available})少于请求的样本数量({num_samples})，将使用所有可用样本")
                num_samples = num_available
            sample_indices = np.random.choice(num_available, num_samples, replace=False)
            print(f"已随机选择 {num_samples} 个样本进行可视化: {sample_indices}")
        else:
            # 验证提供的索引是否有效
            valid_indices = [idx for idx in sample_indices if 0 <= idx < num_available]
            if len(valid_indices) < len(sample_indices):
                print(f"警告: {len(sample_indices) - len(valid_indices)}个样本索引超出范围，将被忽略")
            sample_indices = valid_indices
            if not sample_indices:
                print("错误: 没有有效的样本索引")
                return

        # 确保输出目录存在
        if output_dir is None:
            output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        try:
            # 为每个选定的样本创建力图
            num_plots_created = 0
            for i, idx in enumerate(sample_indices):
                try:
                    if idx >= len(shap_values_clean):
                        print(f"错误: 样本索引 {idx} 超出范围 (0-{len(shap_values_clean) - 1})")
                        continue

                    # 获取单个样本的SHAP值和特征
                    sample_shap = shap_values_clean[idx]
                    sample_features = features_clean[idx]

                    # 检查SHAP值是否有效
                    if np.isnan(sample_shap).any() or np.isinf(sample_shap).any():
                        print(f"警告: 样本 {idx} 的SHAP值包含NaN或Inf，已跳过")
                        continue

                    # 创建力图的简化版本（水平条形图）- 改进的可视化
                    plt.figure(figsize=(12, 8))

                    # 获取特征贡献
                    feature_contribution = sample_shap

                    # 对贡献进行排序，按绝对值大小
                    sorted_idx = np.argsort(np.abs(feature_contribution))[::-1]

                    # 选择前15个特征（或全部，如果少于15个）
                    top_n = min(15, len(sorted_idx))
                    top_idx = sorted_idx[:top_n]

                    # 获取正负贡献
                    contributions = feature_contribution[top_idx]
                    features_names_top = [feature_names[i] for i in top_idx]
                    feature_values = [sample_features[i] for i in top_idx]

                    # 为正负贡献设置不同颜色 - 更美观的配色
                    colors = ['#FF5733' if x < 0 else '#3498DB' for x in contributions]

                    # 创建水平条形图
                    bars = plt.barh(range(len(contributions)), contributions, color=colors,
                                    alpha=0.8, edgecolor='grey', height=0.6)

                    # 设置Y轴标签为特征名称及其值 - 更好的格式化
                    ytick_labels = []
                    for name, value in zip(features_names_top, feature_values):
                        # 根据值的大小选择合适的格式
                        if abs(value) < 0.001 or abs(value) > 1000:
                            formatted_value = f"{value:.2e}"  # 科学计数法
                        else:
                            formatted_value = f"{value:.4g}"  # 通用格式，去除多余的0
                        ytick_labels.append(f"{name} = {formatted_value}")

                    plt.yticks(range(len(contributions)), ytick_labels)

                    # 改进网格
                    plt.grid(axis='x', linestyle='--', alpha=0.4)

                    # 添加零线
                    plt.axvline(x=0, color='black', linestyle='-', alpha=0.7)

                    # 添加每个条形末端的值标签 - 更智能的文本位置
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        # 确定文本位置和颜色
                        if width < 0:
                            text_x = width - max(abs(width) * 0.05, 0.01)
                            ha = 'right'
                            color = 'white' if abs(width) > 0.2 else 'black'
                        else:
                            text_x = width + max(abs(width) * 0.05, 0.01)
                            ha = 'left'
                            color = 'black'

                        # 确定文本y位置和值
                        y_pos = bar.get_y() + bar.get_height() / 2
                        val = round(width, 4)

                        # 添加文本 - 改进的背景和格式
                        plt.text(text_x, y_pos, f"{val:.4f}", va='center', ha=ha, fontsize=9, color=color,
                                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none',
                                           pad=1) if ha == 'left' else None)

                    # 计算预测值（基值+总SHAP值）
                    try:
                        # 尝试获取explainer的预期值作为基值
                        base_value = 0.5  # 默认值
                        predicted_value = base_value + np.sum(feature_contribution)

                        # 添加预测值和基值的标记，使用更美观的布局
                        info_box = plt.text(0.02, 0.97,
                                            f"基值: {base_value:.4f}\n预测值: {predicted_value:.4f}",
                                            transform=plt.gca().transAxes, fontsize=10,
                                            bbox=dict(facecolor='#f8f9fa', alpha=0.9, boxstyle="round,pad=0.5",
                                                      edgecolor='#dee2e6'))
                    except Exception as e:
                        print(f"计算预测值时出错: {e}")

                    # 设置标题和轴标签
                    plt.title(f"样本 #{idx} 的SHAP特征贡献", fontsize=14, pad=20)
                    plt.xlabel("SHAP值（特征贡献）", fontsize=12)

                    # 去除上边框和右边框
                    plt.gca().spines['top'].set_visible(False)
                    plt.gca().spines['right'].set_visible(False)

                    # 保存图表
                    output_path = os.path.join(output_dir, f"{self.dataset_name}_shap_force_plot_sample_{idx}.png")
                    plt.tight_layout()
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()

                    num_plots_created += 1

                except Exception as e:
                    # 单个样本的错误处理
                    print(f"为样本 {idx} 创建SHAP力图时出错: {e}")
                    continue

            # 总结可视化结果
            if num_plots_created > 0:
                print(f"成功创建了 {num_plots_created} 个SHAP力图，保存在 {output_dir}")
            else:
                print("警告: 未能创建任何SHAP力图")

        except Exception as e:
            print(f"创建SHAP力图时出错: {e}")
            import traceback
            traceback.print_exc()

    def visualize_shap_fold_comparison(self, all_fold_shap_values, features, feature_names=None, original_features=None,
                                       max_display=15):
        """
        比较不同折叠之间的SHAP值分布，展示特征重要性的一致性和变化

        参数:
        all_fold_shap_values: 各折的SHAP值字典，格式为 {fold_idx: shap_values_array}
        features: 用于计算重要性的特征数据（通常是测试集特征）
        feature_names: 特征名称列表
        original_features: 原始（未标准化）的特征值，用于显示实际值，如果为None则使用features
        max_display: 最多显示的特征数量
        """
        try:
            # 数据验证
            if not all_fold_shap_values or not isinstance(all_fold_shap_values, dict) or len(all_fold_shap_values) == 0:
                print("错误: all_fold_shap_values必须是非空字典，格式为 {fold_idx: shap_values_array}")
                return

            if features is None:
                print("错误: 必须提供特征数据")
                return

            # 统一数据格式
            if not isinstance(features, np.ndarray):
                features = np.array(features)

            # 使用原始特征（如果提供）
            display_features = original_features if original_features is not None else features
            if not isinstance(display_features, np.ndarray):
                display_features = np.array(display_features)

            # 确保有特征名称
            if feature_names is None:
                feature_names = [f"特征_{i}" for i in range(features.shape[1])]
            elif len(feature_names) != features.shape[1]:
                print(f"警告: 特征名称数量({len(feature_names)})与特征数量({features.shape[1]})不匹配")
                feature_names = [f"特征_{i}" for i in range(features.shape[1])]

            # 获取所有折
            fold_indices = sorted(all_fold_shap_values.keys())
            n_folds = len(fold_indices)

            print(f"分析 {n_folds} 个折叠的SHAP值...")

            # 计算每个折叠的特征重要性（平均绝对SHAP值）
            fold_importances = {}
            all_importances = np.zeros((n_folds, features.shape[1]))

            for i, fold_idx in enumerate(fold_indices):
                shap_values = all_fold_shap_values[fold_idx]
                # 处理和清理数据
                if not isinstance(shap_values, np.ndarray):
                    shap_values = np.array(shap_values)
                shap_values = np.nan_to_num(shap_values)

                # 计算特征重要性
                importance = np.abs(shap_values).mean(0)
                fold_importances[fold_idx] = importance
                all_importances[i] = importance

            # 计算所有折叠的平均特征重要性
            mean_importance = np.mean(all_importances, axis=0)
            std_importance = np.std(all_importances, axis=0)

            # 根据平均重要性排序特征
            sorted_idx = np.argsort(-mean_importance)
            top_idx = sorted_idx[:max_display]

            # 创建折叠间特征重要性比较图
            plt.figure(figsize=(14, 10))

            # 准备条形图数据
            bar_positions = np.arange(max_display)
            bar_width = 0.7 / n_folds

            # 使用颜色映射来区分不同的折叠
            colors = plt.cm.viridis(np.linspace(0, 1, n_folds))

            # 绘制每个折叠的特征重要性
            for i, fold_idx in enumerate(fold_indices):
                importance = fold_importances[fold_idx]
                plt.bar(bar_positions + i * bar_width - 0.35 + bar_width / 2,
                        importance[top_idx],
                        width=bar_width,
                        alpha=0.7,
                        color=colors[i],
                        label=f'折叠 {fold_idx}')

            # 在右侧添加平均值和标准差
            for i, idx in enumerate(top_idx):
                # 平均值
                plt.plot([bar_positions[i] - 0.35, bar_positions[i] + 0.35],
                         [mean_importance[idx], mean_importance[idx]],
                         'k-', linewidth=2)

                # 标准差上下限
                plt.plot([bar_positions[i], bar_positions[i]],
                         [mean_importance[idx] - std_importance[idx], mean_importance[idx] + std_importance[idx]],
                         'k-', linewidth=1.5)

            # 设置图表标签和属性
            plt.xlabel('平均|SHAP值|（特征重要性）', fontsize=12)
            plt.title('各折叠的SHAP特征重要性比较', fontsize=14)
            plt.xticks(bar_positions, [feature_names[i] for i in top_idx], rotation=45, ha='right')
            plt.legend(title='折叠')
            plt.grid(True, axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()

            # 保存图片
            plt.savefig(os.path.join(self.output_dir, f"{self.dataset_name}_shap_fold_comparison_bar.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()

            # 创建箱线图，显示每个特征在不同折叠间的重要性分布
            plt.figure(figsize=(14, 8))

            # 准备箱线图数据
            box_data = []
            for i, idx in enumerate(top_idx):
                # 获取特定特征在所有折叠中的重要性
                feature_importances = [fold_importances[fold_idx][idx] for fold_idx in fold_indices]
                box_data.append(feature_importances)

            # 绘制箱线图
            boxplots = plt.boxplot(box_data, vert=False, patch_artist=True,
                                   labels=[feature_names[i] for i in top_idx])

            # 自定义箱线图颜色
            for box in boxplots['boxes']:
                box.set(facecolor='#ADD8E6', alpha=0.7)

            # 添加散点显示各折的值
            for i, importances in enumerate(box_data):
                y_pos = i + 1  # 箱线图位置从1开始
                x_positions = importances

                # 使用与前一个图相同的颜色映射
                for j, x_pos in enumerate(x_positions):
                    plt.scatter(x_pos, y_pos, color=colors[j], s=50, alpha=0.8,
                                edgecolor='black', linewidth=0.5)

            # 设置图表属性
            plt.title('特征重要性在各折叠间的分布', fontsize=14)
            plt.xlabel('平均|SHAP值|', fontsize=12)
            plt.grid(True, axis='x', linestyle='--', alpha=0.3)
            plt.tight_layout()

            # 保存图片
            plt.savefig(os.path.join(self.output_dir, f"{self.dataset_name}_shap_fold_comparison_box.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()

            # 创建热图，展示每个折叠中前N个特征的变化
            plt.figure(figsize=(12, 8))

            # 准备热图数据
            heatmap_data = np.zeros((n_folds, max_display))
            for i, fold_idx in enumerate(fold_indices):
                importance = fold_importances[fold_idx]
                heatmap_data[i] = importance[top_idx]

            # 绘制热图
            im = plt.imshow(heatmap_data, aspect='auto', cmap='viridis')

            # 添加颜色条
            cbar = plt.colorbar(im)
            cbar.set_label('平均|SHAP值|', rotation=270, labelpad=20)

            # 设置轴标签
            plt.yticks(np.arange(n_folds), [f'折叠 {fold_idx}' for fold_idx in fold_indices])
            plt.xticks(np.arange(max_display), [feature_names[i] for i in top_idx], rotation=45, ha='right')

            # 在每个单元格中标注值
            for i in range(n_folds):
                for j in range(max_display):
                    plt.text(j, i, f"{heatmap_data[i, j]:.4f}",
                             ha="center", va="center",
                             color="white" if heatmap_data[i, j] > np.max(heatmap_data) / 2 else "black",
                             fontsize=8)

            plt.title('各折叠中的特征重要性热图', fontsize=14)
            plt.tight_layout()

            # 保存图片
            plt.savefig(os.path.join(self.output_dir, f"{self.dataset_name}_shap_fold_comparison_heatmap.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()

            # 创建相关性图，比较不同折叠之间的特征重要性相关性
            if n_folds > 2:  # 只有当有超过2个折叠时才有意义
                plt.figure(figsize=(10, 8))

                # 计算相关性矩阵
                importance_corr = np.corrcoef(all_importances)

                # 绘制热图
                im = plt.imshow(importance_corr, cmap='coolwarm', vmin=-1, vmax=1)

                # 添加颜色条
                cbar = plt.colorbar(im)
                cbar.set_label('相关系数', rotation=270, labelpad=20)

                # 设置轴标签
                plt.xticks(np.arange(n_folds), [f'折叠 {fold_idx}' for fold_idx in fold_indices])
                plt.yticks(np.arange(n_folds), [f'折叠 {fold_idx}' for fold_idx in fold_indices])

                # 在每个单元格中标注相关系数
                for i in range(n_folds):
                    for j in range(n_folds):
                        plt.text(j, i, f"{importance_corr[i, j]:.2f}",
                                 ha="center", va="center",
                                 color="white" if abs(importance_corr[i, j]) > 0.5 else "black")

                plt.title('折叠间特征重要性相关性矩阵', fontsize=14)
                plt.tight_layout()

                # 保存图片
                plt.savefig(os.path.join(self.output_dir, f"{self.dataset_name}_shap_fold_correlation.png"),
                            dpi=300, bbox_inches='tight')
                plt.close()

            # 计算各折叠中的特征排名并比较排名的一致性
            ranks = np.zeros((n_folds, features.shape[1]))
            for i, fold_idx in enumerate(fold_indices):
                importance = fold_importances[fold_idx]
                # 计算排名，排名相同时取平均值
                ranks[i] = features.shape[1] - stats.rankdata(importance, method='average') + 1

            # 计算肯德尔W一致性系数 - 衡量排名一致性
            try:
                from scipy.stats import kendalltau
                # 计算每对折叠之间的Kendall's tau
                kendall_taus = []
                for i in range(n_folds):
                    for j in range(i + 1, n_folds):
                        tau, _ = kendalltau(ranks[i], ranks[j])
                        kendall_taus.append(tau)

                avg_kendall_tau = np.mean(kendall_taus)

                # 计算特征重要性排名的一致性报告
                top_feature_agreement = {}

                # 计算各折叠中前N个特征的重叠数量
                for n in [5, 10, max_display]:
                    n = min(n, features.shape[1])
                    fold_top_features = []
                    for i, fold_idx in enumerate(fold_indices):
                        importance = fold_importances[fold_idx]
                        top_n_idx = np.argsort(-importance)[:n]
                        fold_top_features.append(set(top_n_idx))

                    # 计算重叠比例
                    common_features = set.intersection(*fold_top_features)
                    top_feature_agreement[n] = {
                        "overlap_count": len(common_features),
                        "overlap_ratio": len(common_features) / n,
                        "common_features": [feature_names[i] for i in common_features]
                    }

                # 将结果保存到文本文件
                with open(os.path.join(self.output_dir, f"{self.dataset_name}_shap_consistency_report.txt"), 'w') as f:
                    f.write(f"SHAP特征重要性一致性报告\n")
                    f.write(f"==========================\n\n")
                    f.write(f"分析了 {n_folds} 个折叠的SHAP值\n\n")

                    f.write(f"排名一致性:\n")
                    f.write(f"平均Kendall's tau: {avg_kendall_tau:.4f} (值越接近1表示排名越一致)\n\n")

                    f.write(f"前N个特征的重叠情况:\n")
                    for n, agreement in top_feature_agreement.items():
                        f.write(
                            f"前{n}个特征: {agreement['overlap_count']}/{n} 重叠 ({agreement['overlap_ratio'] * 100:.1f}%)\n")
                        if agreement['common_features']:
                            f.write(f"  共同特征: {', '.join(agreement['common_features'])}\n")
                        else:
                            f.write(f"  没有共同特征\n")
                        f.write("\n")

                    # 添加各折的Top 10特征
                    f.write(f"各折叠的前10个特征:\n")
                    for i, fold_idx in enumerate(fold_indices):
                        importance = fold_importances[fold_idx]
                        top_n_idx = np.argsort(-importance)[:10]
                        f.write(f"折叠 {fold_idx}:\n")
                        for rank, idx in enumerate(top_n_idx):
                            f.write(f"  {rank + 1}. {feature_names[idx]} ({importance[idx]:.4f})\n")
                        f.write("\n")

                print(
                    f"SHAP特征重要性一致性报告已保存到 {self.output_dir}/{self.dataset_name}_shap_consistency_report.txt")

            except Exception as e:
                print(f"计算特征排名一致性时出错: {e}")
                import traceback
                traceback.print_exc()

            print(f"SHAP折叠比较分析完成，已生成图表保存在 {self.output_dir}")

        except Exception as e:
            print(f"创建SHAP折叠比较图时出错: {e}")
            import traceback
            traceback.print_exc()
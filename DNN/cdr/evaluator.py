import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import os
from config import Config
from tqdm import tqdm


class ModelEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def evaluate(self, test_loader):
        """评估模型性能"""
        predictions = []
        actuals = []

        with torch.no_grad():
            for features, targets in tqdm(test_loader, desc="评估中"):
                features, targets = features.to(self.device), targets.to(self.device)
                outputs = self.model(features)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.cpu().numpy())

        predictions = np.array(predictions).flatten()
        actuals = np.array(actuals).flatten()

        # 计算评估指标
        metrics = {
            'R²': r2_score(actuals, predictions),
            'RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
            'MAE': mean_absolute_error(actuals, predictions),
            '平均相对误差(%)': np.mean(np.abs((predictions - actuals) / actuals * 100)),
            '最大相对误差(%)': np.max(np.abs((predictions - actuals) / actuals * 100)),
            '最小相对误差(%)': np.min(np.abs((predictions - actuals) / actuals * 100))
        }

        # 创建详细的结果DataFrame
        results_df = pd.DataFrame({
            '实际值': actuals,
            '预测值': predictions,
            '绝对误差': np.abs(predictions - actuals),
            '相对误差(%)': np.abs((predictions - actuals) / actuals * 100)
        })

        # 保存结果
        output_dir = Config.TEST_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        # 保存详细结果
        results_df.to_csv(os.path.join(output_dir, f"{Config.DATASET_NAME}_detailed_results.csv"), index=False)

        # 保存评估指标
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(output_dir, f"{Config.DATASET_NAME}_metrics.csv"), index=False)

        # 打印评估结果
        print("\n=== 模型评估结果 ===")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        return predictions, actuals, metrics

    def evaluate_train_val(self, train_loader, val_loader):
        """评估训练集和验证集的性能"""
        train_predictions, train_actuals, train_metrics = self._evaluate_loader(train_loader, "训练集")
        val_predictions, val_actuals, val_metrics = self._evaluate_loader(val_loader, "验证集")

        return {
            'train': {'predictions': train_predictions, 'actuals': train_actuals, 'metrics': train_metrics},
            'val': {'predictions': val_predictions, 'actuals': val_actuals, 'metrics': val_metrics}
        }

    def _evaluate_loader(self, loader, set_name):
        """评估单个数据加载器的性能"""
        predictions = []
        actuals = []

        with torch.no_grad():
            for features, targets in loader:
                features, targets = features.to(self.device), targets.to(self.device)
                outputs = self.model(features)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.cpu().numpy())

        predictions = np.array(predictions).flatten()
        actuals = np.array(actuals).flatten()

        metrics = {
            'R²': r2_score(actuals, predictions),
            'RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
            'MAE': mean_absolute_error(actuals, predictions),
            '平均相对误差(%)': np.mean(np.abs((predictions - actuals) / actuals * 100))
        }

        print(f"\n=== {set_name}评估结果 ===")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        return predictions, actuals, metrics

    @staticmethod
    def evaluate_cv_results(all_predictions, all_actuals, fold_metrics=None):
        """
        评估交叉验证的总体结果

        参数:
        all_predictions: 所有折叠的预测值
        all_actuals: 所有折叠的实际值
        fold_metrics: 每个折叠的指标，格式为[{fold1_metrics}, {fold2_metrics}, ...]

        返回:
        overall_metrics: 所有折叠合并后的整体指标
        avg_metrics: 各折叠指标的平均值
        """
        # 计算整体指标
        overall_metrics = self.calculate_metrics(all_actuals, all_predictions)

        # 如果提供了每折的指标，计算平均值
        avg_metrics = {}
        cv_metrics = {}

        if fold_metrics:
            # 提取每个指标
            for metric in ['r2', 'rmse', 'mae']:
                # 将指标名称格式化为显示格式
                display_name = 'R²' if metric == 'r2' else metric.upper()
                metric_values = [fold[metric] for fold in fold_metrics]
                cv_metrics[display_name] = metric_values
                avg_metrics[f'avg_{metric}'] = np.mean(metric_values)
                avg_metrics[f'std_{metric}'] = np.std(metric_values)

        # 保存整体评估结果
        self.save_metrics(overall_metrics, 'overall_metrics')

        # 如果有每折指标，保存平均值
        if fold_metrics:
            self.save_metrics(avg_metrics, 'avg_metrics')

            # 创建每折指标对比表格
            fold_comparison_df = pd.DataFrame({
                '折叠': [f'折叠_{i + 1}' for i in range(len(fold_metrics))],
                'R²': [fold['r2'] for fold in fold_metrics],
                'RMSE': [fold['rmse'] for fold in fold_metrics],
                'MAE': [fold['mae'] for fold in fold_metrics]
            })

            # 添加平均行
            avg_row = pd.DataFrame({
                '折叠': ['平均值'],
                'R²': [avg_metrics['avg_r2']],
                'RMSE': [avg_metrics['avg_rmse']],
                'MAE': [avg_metrics['avg_mae']]
            })

            fold_comparison_df = pd.concat([fold_comparison_df, avg_row], ignore_index=True)
            fold_comparison_df.to_csv(os.path.join(self.output_dir, f'{self.dataset_name}_fold_comparison.csv'),
                                      index=False)

        return overall_metrics, avg_metrics, cv_metrics

    def calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r = np.corrcoef(y_true, y_pred)[0, 1]  # 计算相关系数r

        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'r': r
        }

    def calculate_metrics_history(self, y_true, y_pred_history):
        """
        计算指标历史记录

        参数：
        y_true: 实际值
        y_pred_history: 训练过程中每个epoch的预测值历史记录 shape=[epochs, samples]

        返回：
        metrics_history: 包含每个epoch的评估指标
        """
        metrics_history = {
            'r2_history': [],
            'rmse_history': [],
            'mae_history': []
        }

        for epoch_preds in y_pred_history:
            # 计算当前epoch的指标
            metrics = self.calculate_metrics(y_true, epoch_preds)

            # 保存指标历史
            metrics_history['r2_history'].append(metrics['r2'])
            metrics_history['rmse_history'].append(metrics['rmse'])
            metrics_history['mae_history'].append(metrics['mae'])

        return metrics_history

    def save_metrics(self, metrics, filename_prefix):
        """保存评估指标到CSV文件"""
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(self.output_dir, f'{self.dataset_name}_{filename_prefix}.csv'), index=False)

    def evaluate_fold(self, y_true, y_pred, fold_idx, pred_history=None):
        """
        评估单个折叠的结果

        参数:
        y_true: 实际值
        y_pred: 预测值
        fold_idx: 折叠索引
        pred_history: 可选，每个epoch的预测历史记录

        返回:
        fold_metrics: 包含该折叠评估指标的字典
        """
        # 计算基本指标
        metrics = self.calculate_metrics(y_true, y_pred)

        # 如果有预测历史，计算指标历史
        if pred_history is not None:
            history_metrics = self.calculate_metrics_history(y_true, pred_history)
            # 合并基本指标和历史指标
            metrics.update(history_metrics)

        # 保存当前折叠的指标
        self.save_metrics(metrics, f'fold_{fold_idx}_metrics')

        return metrics
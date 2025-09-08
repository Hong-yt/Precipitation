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
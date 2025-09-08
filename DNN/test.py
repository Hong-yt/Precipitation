import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from config import Config
from dataset import PrecipitationDataset
from model import PrecipitationDNN
from evaluator import ModelEvaluator
from visualizer import ResultVisualizer
from utils import get_device, load_checkpoint


def test(model_path, scaler_path, test_data_path):
    """
    测试模型性能

    参数:
    model_path: 模型文件路径
    scaler_path: 标准化器文件路径
    test_data_path: 测试数据文件路径
    """
    # 设置设备
    device = get_device()

    # 加载测试数据
    test_data = pd.read_csv(test_data_path)
    X_test = test_data.drop('target', axis=1).values
    y_test = test_data['target'].values

    # 加载标准化器
    scaler = torch.load(scaler_path)

    # 创建测试数据集和数据加载器
    test_dataset = PrecipitationDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE
    )

    # 初始化模型
    model = PrecipitationDNN(
        input_size=X_test.shape[1],
        hidden_sizes=Config.HIDDEN_SIZES,
        dropout_rate=Config.DROPOUT_RATE
    ).to(device)

    # 加载模型权重
    model = load_checkpoint(model, model_path, device)

    # 评估模型
    evaluator = ModelEvaluator(model, device)
    predictions, actuals, metrics = evaluator.evaluate(test_loader)

    # 可视化结果
    visualizer = ResultVisualizer(
        dataset_name=Config.DATASET_NAME,
        output_dir=Config.TEST_OUTPUT_DIR
    )
    visualizer.visualize_predictions(predictions, actuals)

    # 保存评估结果
    # 保存预测结果
    results_df = pd.DataFrame({
        'actual': actuals,
        'predicted': predictions,
        'error': predictions - actuals
    })
    results_df.to_csv(os.path.join(Config.TEST_OUTPUT_DIR, f"{Config.DATASET_NAME}_predictions.csv"), index=False)

    # 保存评估指标
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(Config.TEST_OUTPUT_DIR, f"{Config.DATASET_NAME}_metrics.csv"), index=False)

    return metrics


if __name__ == "__main__":
    # 设置文件路径
    model_path = os.path.join(
        Config.TRAIN_OUTPUT_DIR,
        f"{Config.DATASET_NAME}_model.pth"
    )
    scaler_path = os.path.join(
        Config.TRAIN_OUTPUT_DIR,
        f"{Config.DATASET_NAME}_scaler.pth"
    )
    test_data_path = os.path.join(
        Config.TRAIN_OUTPUT_DIR,
        f"{Config.DATASET_NAME}_test.csv"
    )

    # 运行测试
    metrics = test(model_path, scaler_path, test_data_path)

    # 打印评估指标
    print("\n=== 测试集评估指标 ===")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")
import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from torch.utils.data import DataLoader

from config import Config
from dataset import PrecipitationDataset, PrecipitationDataProcessor
from model import PrecipitationDNN
from utils import get_device, load_checkpoint


def load_model(fold_idx, input_size):
    """加载指定折的模型"""
    model_path = os.path.join(Config.CV_OUTPUT_DIR, f"{Config.DATASET_NAME}_fold{fold_idx}_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 使用与训练时完全相同的隐藏层配置
    hidden_sizes = [256, 128, 64, 32]  # 与训练时保持一致

    model = PrecipitationDNN.create_model_for_fold(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        dropout_rate=Config.DROPOUT_RATE,
        fold_idx=fold_idx,
        activation=Config.ACTIVATION
    )

    # 加载完整的检查点
    checkpoint = torch.load(model_path)
    # 只加载模型状态字典
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def load_scaler(fold_idx):
    """加载指定折的标准化器"""
    scaler_path = os.path.join(Config.CV_OUTPUT_DIR, f"{Config.DATASET_NAME}_fold{fold_idx}_scaler.pth")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"标准化器文件不存在: {scaler_path}")

    return torch.load(scaler_path)


def apply_feature_engineering(X):
    """
    对新数据应用与训练数据相同的特征工程处理

    参数:
    X: 原始特征矩阵

    返回:
    X_engineered: 处理后的特征矩阵
    """
    print("\n=== 应用特征工程 ===")

    # 1. 非线性变换
    if Config.NONLINEAR_TRANSFORM:
        print("应用非线性变换...")
        X_sqrt = np.sqrt(np.abs(X))
        X_square = np.power(X, 2)
        X_log = np.log1p(np.abs(X))
        X = np.hstack((X, X_sqrt, X_square, X_log))
        print(f"非线性变换后特征数: {X.shape[1]}")

    # 2. 交互特征
    if Config.INTERACTION_FEATURES:
        print("创建交互特征...")
        poly = PolynomialFeatures(degree=Config.INTERACTION_DEGREE, include_bias=False)
        X = poly.fit_transform(X)
        print(f"交互特征后特征数: {X.shape[1]}")

    # 3. 统计特征
    if Config.STATISTICAL_FEATURES:
        print("创建统计特征...")
        row_mean = np.mean(X, axis=1).reshape(-1, 1)
        row_std = np.std(X, axis=1).reshape(-1, 1)
        row_min = np.min(X, axis=1).reshape(-1, 1)
        row_max = np.max(X, axis=1).reshape(-1, 1)
        row_median = np.median(X, axis=1).reshape(-1, 1)
        row_q25 = np.percentile(X, 25, axis=1).reshape(-1, 1)
        row_q75 = np.percentile(X, 75, axis=1).reshape(-1, 1)
        row_range = row_max - row_min
        row_iqr = row_q75 - row_q25

        X = np.hstack((
            X,
            row_mean, row_std, row_min, row_max, row_median,
            row_q25, row_q75, row_range, row_iqr
        ))
        print(f"统计特征后特征数: {X.shape[1]}")

    return X


def predict_new_data(model_path, scaler_path, new_data_path, output_path):
    """
    使用训练好的模型对新数据进行预测

    参数:
    model_path: 模型文件路径
    scaler_path: 标准化器文件路径
    new_data_path: 新数据文件路径（xlsx格式）
    output_path: 预测结果保存路径
    """
    # 设置设备
    device = get_device()

    # 加载新数据
    print("加载新数据...")
    print(f"尝试读取文件: {new_data_path}")
    print(f"文件是否存在: {os.path.exists(new_data_path)}")
    try:
        # 读取所有数据
        full_data = pd.read_csv(new_data_path)
        print("成功读取数据文件")
        print(f"完整数据形状: {full_data.shape}")

        # 保存行列号
        row_col = full_data.iloc[:, :2]

        # 只使用特征列（排除前两列）
        new_data = full_data.iloc[:, 2:]
        print(f"特征数据形状: {new_data.shape}")
        print(f"特征数量: {len(new_data.columns)}")
        print(f"特征名称: {list(new_data.columns)}")
    except Exception as e:
        print(f"读取数据文件时出错: {str(e)}")
        raise

    # 应用特征工程
    X_new = new_data.values
    X_new = apply_feature_engineering(X_new)
    print(f"特征工程后的特征数量: {X_new.shape[1]}")

    # 加载标准化器
    print("加载标准化器...")
    try:
        scaler = torch.load(scaler_path)
        print(f"标准化器期望的特征数量: {len(scaler.mean_)}")
    except Exception as e:
        print(f"加载标准化器时出错: {str(e)}")
        raise

    # 标准化数据
    print("标准化数据...")
    try:
        X_new_scaled = scaler.transform(X_new)
        print("数据标准化成功")
    except Exception as e:
        print(f"数据标准化时出错: {str(e)}")
        raise

    # 创建数据集和数据加载器
    dataset = PrecipitationDataset(X_new_scaled, np.zeros(len(X_new_scaled)))
    data_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE)

    # 加载模型
    print("加载模型...")
    model = PrecipitationDNN(
        input_size=X_new_scaled.shape[1],
        hidden_sizes=[256, 128, 64, 32],  # 使用与训练时相同的配置
        dropout_rate=Config.DROPOUT_RATE,
        activation=Config.ACTIVATION
    )
    model = load_checkpoint(model, model_path, device)
    model = model.to(device)  # 确保模型在正确的设备上
    model.eval()

    # 进行预测
    print("开始预测...")
    predictions = []
    with torch.no_grad():
        for features, _ in data_loader:
            features = features.to(device)  # 将数据移动到正确的设备上
            outputs = model(features)
            predictions.extend(outputs.cpu().numpy())

    predictions = np.array(predictions).flatten()

    # 打印预测值的统计信息
    print("\n=== 预测值统计信息（逆变换前）===")
    print(f"最小值: {np.min(predictions):.4f}")
    print(f"最大值: {np.max(predictions):.4f}")
    print(f"平均值: {np.mean(predictions):.4f}")
    print(f"标准差: {np.std(predictions):.4f}")
    print(f"负值数量: {np.sum(predictions < 0)}")
    print(f"负值比例: {np.sum(predictions < 0) / len(predictions) * 100:.2f}%")

    # 如果训练时应用了对数变换，需要进行逆变换
    if Config.LOG_TRANSFORM:
        print("\n=== 应用目标变量的逆变换 ===")
        print(f"使用偏移量: {Config.LOG_OFFSET}")
        predictions = np.exp(predictions) - Config.LOG_OFFSET

        # 打印逆变换后的统计信息
        print("\n=== 预测值统计信息（逆变换后）===")
        print(f"最小值: {np.min(predictions):.4f}")
        print(f"最大值: {np.max(predictions):.4f}")
        print(f"平均值: {np.mean(predictions):.4f}")
        print(f"标准差: {np.std(predictions):.4f}")
        print(f"负值数量: {np.sum(predictions < 0)}")
        print(f"负值比例: {np.sum(predictions < 0) / len(predictions) * 100:.2f}%")

    # 保存预测结果
    print("\n保存预测结果...")
    results_df = pd.DataFrame({
        '行号': row_col.iloc[:, 0],
        '列号': row_col.iloc[:, 1],
        '预测值': predictions
    })
    results_df.to_excel(output_path, index=False)
    print(f"预测结果已保存到: {output_path}")


if __name__ == "__main__":
    # 设置路径
    model_path = os.path.join(Config.CV_OUTPUT_DIR, f"{Config.DATASET_NAME}_fold1_model.pth")  # 使用第一折的模型
    scaler_path = os.path.join(Config.CV_OUTPUT_DIR, f"{Config.DATASET_NAME}_fold1_scaler.pth")  # 使用第一折的标准化器
    new_data_path = "D:/hongyouting/data/dnn/merge/features/2/imerg_f7_features.csv"  # 请替换为实际的新数据文件路径
    output_path = "D:/hongyouting/data/dnn/merge/features/2/f7_predictions.xlsx"

    # 执行预测
    predict_new_data(model_path, scaler_path, new_data_path, output_path)
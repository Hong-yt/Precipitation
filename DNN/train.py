import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from config import Config
from dataset import PrecipitationDataProcessor, PrecipitationDataset
from model import PrecipitationDNN
from trainer import PrecipitationModelTrainer
from visualizer import ResultVisualizer
from utils import set_seed, get_device, save_experiment_results


def train():
    """训练模型并保存训练结果"""
    print("\n=== 开始训练流程 ===")

    # 设置随机种子和设备
    set_seed(Config.TORCH_SEED, Config.NUMPY_SEED)
    device = get_device()

    # 创建输出目录
    Config.create_directories()

    # 数据加载和预处理
    print("\n=== 数据处理 ===")
    data_processor = PrecipitationDataProcessor(
        Config.DATASET_NAME,
        Config.FEATURES_PATH,
        Config.TARGET_PATH
    )

    features_df, target_df = data_processor.load_data()
    X, y = data_processor.preprocess_data()

    # 划分数据集
    X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data(
        X, y,
        test_size=Config.TEST_SIZE,
        val_size=Config.VAL_SIZE
    )

    print(f"数据集大小: 训练集 {len(X_train)}, 验证集 {len(X_val)}, 测试集 {len(X_test)}")

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 保存数据集划分结果
    train_data = pd.DataFrame(X_train)
    train_data['target'] = y_train
    train_data.to_csv(os.path.join(Config.TRAIN_OUTPUT_DIR, f"{Config.DATASET_NAME}_train.csv"), index=False)

    val_data = pd.DataFrame(X_val)
    val_data['target'] = y_val
    val_data.to_csv(os.path.join(Config.TRAIN_OUTPUT_DIR, f"{Config.DATASET_NAME}_val.csv"), index=False)

    test_data = pd.DataFrame(X_test)
    test_data['target'] = y_test
    test_data.to_csv(os.path.join(Config.TRAIN_OUTPUT_DIR, f"{Config.DATASET_NAME}_test.csv"), index=False)

    # 保存标准化器
    scaler_path = os.path.join(Config.TRAIN_OUTPUT_DIR, f"{Config.DATASET_NAME}_scaler.pth")
    torch.save(scaler, scaler_path)

    # 创建数据集和数据加载器
    train_dataset = PrecipitationDataset(X_train, y_train)
    val_dataset = PrecipitationDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE
    )

    # 初始化模型
    print("\n=== 初始化模型 ===")
    model = PrecipitationDNN(
        input_size=X_train.shape[1],
        hidden_sizes=Config.HIDDEN_SIZES,
        dropout_rate=Config.DROPOUT_RATE
    ).to(device)

    # 初始化训练器
    trainer = PrecipitationModelTrainer(
        model=model,
        dataset_name=Config.DATASET_NAME,
        device=device,
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    # 训练模型
    print("\n=== 开始训练模型 ===")
    train_losses, val_losses = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.EPOCHS,
        patience=Config.PATIENCE
    )

    # 保存训练结果
    # 保存损失历史
    loss_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    loss_df.to_csv(os.path.join(Config.TRAIN_OUTPUT_DIR, f"{Config.DATASET_NAME}_losses.csv"), index=False)

    # 保存最佳模型
    best_model_path = os.path.join(Config.TRAIN_OUTPUT_DIR, f"{Config.DATASET_NAME}_model.pth")
    torch.save(model.state_dict(), best_model_path)
    print(f"\n最佳模型已保存到: {best_model_path}")

    # 可视化训练结果
    print("\n=== 生成可视化结果 ===")
    visualizer = ResultVisualizer(
        dataset_name=Config.DATASET_NAME,
        output_dir=Config.TRAIN_OUTPUT_DIR
    )
    visualizer.visualize_losses(train_losses, val_losses)

    print("\n=== 训练完成 ===")
    return model, scaler, (X_test, y_test)


if __name__ == "__main__":
    train()
import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd

from config import Config
from dataset import PrecipitationDataProcessor, PrecipitationDataset
from model import PrecipitationDNN
from trainer import PrecipitationModelTrainer
from evaluator import ModelEvaluator
from visualizer import ResultVisualizer
from utils import set_seed, get_device, save_experiment_results, cross_validation_on_train_set
from torch.utils.data import DataLoader


def main():
    # 设置随机种子和设备
    set_seed(Config.SEED)
    device = get_device()

    # 创建输出目录
    Config.create_directories()

    # 数据加载和预处理
    data_processor = PrecipitationDataProcessor(
        Config.DATASET_NAME,
        Config.FEATURES_PATH,
        Config.TARGET_PATH
    )

    # 加载数据
    features_df, target_df = data_processor.load_data()

    # 预处理数据，但不进行数据增强（只做基本预处理和缺失值处理）
    X, y = data_processor.preprocess_data(
        add_poly_features=False,  # 暂不添加多项式特征
        add_interactions=False,  # 暂不添加特征交互项
        augment_data=False  # 暂不添加数据增强
    )

    # 首先划分为训练+验证集 vs 测试集，确保测试集不被增强
    X_train_val, X_test, y_train_val, y_test = data_processor.split_test_set(
        X, y, test_size=Config.TEST_SIZE
    )

    # 对训练+验证集进行特征扩展和数据增强
    print("\n=== 对训练+验证集进行特征工程 ===")

    # 注意：首先进行标准化，然后再添加多项式特征
    scaler = StandardScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val)

    # 将标准化后的数据传递给特征工程
    X_train_val_processed = data_processor.feature_engineering(
        X_train_val_scaled,
        add_poly_features=True,  # 保留多项式特征但会限制阶数
        add_interactions=False  # 不添加交互特征以减少特征数量
    )

    # 保存标准化器，以便在测试集上使用相同的变换
    data_processor.scaler = scaler

    # 进行数据增强
    print("\n=== 对训练+验证集进行数据增强 ===")
    X_train_val_augmented, y_train_val_augmented = data_processor.augment_with_noise(
        X_train_val_processed, y_train_val, noise_level=0.015  # 减少噪声水平
    )

    # 确保测试集经过相同的特征变换（但不进行增强）
    print("\n=== 对测试集应用相同的特征工程（不增强）===")
    X_test_processed = data_processor.transform_test_data(X_test)  # 使用与训练集相同的特征工程
    print(f"测试集特征工程: 原始特征数 {X_test.shape[1]}")
    print(f"处理后测试集特征数: {X_test_processed.shape[1]}")

    # 此时只划分训练集和验证集用于交叉验证
    # 注意：此时不对数据进行标准化，交叉验证会在每一折中独立标准化
    X_train_cv, X_val_cv, y_train_cv, y_val_cv = data_processor.split_train_val(
        X_train_val_processed, y_train_val, val_size=Config.VAL_SIZE
    )

    # 创建交叉验证数据集
    print("\n=== 开始交叉验证 ===")
    cv_results = cross_validation_on_train_set(
        X_train_cv, y_train_cv,
        model_class=PrecipitationDNN,
        n_splits=Config.N_SPLITS,
        epochs=Config.CV_EPOCHS,
        device=device,
        hidden_sizes=Config.HIDDEN_SIZES,
        weight_decay=Config.WEIGHT_DECAY,
        patience=Config.PATIENCE
    )

    # 从交叉验证结果中获取最佳超参数
    optimal_batch_size = cv_results['best_batch_size']
    optimal_dropout = cv_results['best_dropout']
    optimal_lr = cv_results['best_lr']

    print(f"\n交叉验证找到的最佳超参数:")
    print(f"Batch Size: {optimal_batch_size}")
    print(f"Dropout Rate: {optimal_dropout}")
    print(f"Learning Rate: {optimal_lr}")

    # 对全部训练+验证数据应用相同的数据增强和处理
    print("\n=== 为最终训练准备数据 ===")

    # 首先对全部训练+验证数据进行数据增强
    X_train_val_final, y_train_val_final = data_processor.augment_with_noise(
        X_train_val_processed, y_train_val, noise_level=0.015  # 使用相同的噪声水平
    )

    # 再划分训练+验证集为训练集和验证集
    X_train, X_val, y_train, y_val = data_processor.split_train_val(
        X_train_val_final, y_train_val_final, val_size=Config.VAL_SIZE
    )

    # 标准化特征
    # 注意：使用全部训练集数据拟合标准化器
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test_processed)

    print(f"最终数据集大小:")
    print(f"训练集: {X_train_scaled.shape}, {y_train.shape}")
    print(f"验证集: {X_val_scaled.shape}, {y_val.shape}")
    print(f"测试集: {X_test_scaled.shape}, {y_test.shape}")

    # 创建数据集和数据加载器（最终训练）
    train_dataset = PrecipitationDataset(X_train_scaled, y_train)
    val_dataset = PrecipitationDataset(X_val_scaled, y_val)
    test_dataset = PrecipitationDataset(X_test_scaled, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=optimal_batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=optimal_batch_size
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=optimal_batch_size
    )

    # 初始化最终模型（使用交叉验证的最佳配置）
    print(f"\n使用交叉验证找到的最佳模型配置进行最终训练")

    # 初始化模型，使用最佳超参数
    model = PrecipitationDNN(
        input_size=X_train_scaled.shape[1],
        hidden_sizes=Config.HIDDEN_SIZES,
        dropout_rate=optimal_dropout
    )
    model.to(device)

    # 初始化训练器，使用交叉验证找到的最佳学习率
    trainer = PrecipitationModelTrainer(
        model=model,
        dataset_name=Config.DATASET_NAME,
        device=device,
        learning_rate=optimal_lr,
        weight_decay=Config.WEIGHT_DECAY
    )

    # 训练最终模型
    print("\n=== 开始最终模型训练 ===")
    train_losses, val_losses = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.EPOCHS
    )

    # 保存训练结果
    train_results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'hyperparams': {
            'dropout_rate': optimal_dropout,
            'learning_rate': optimal_lr,
            'batch_size': optimal_batch_size
        }
    }
    save_experiment_results(
        train_results,
        Config.TRAIN_OUTPUT_DIR,
        f"{Config.DATASET_NAME}_train"
    )

    # 可视化训练结果
    train_visualizer = ResultVisualizer(
        dataset_name=Config.DATASET_NAME,
        output_dir=Config.TRAIN_OUTPUT_DIR
    )
    train_visualizer.visualize_losses(train_losses, val_losses)

    # 评估模型在所有数据集上的表现
    print("\n=== 开始全面模型评估 ===")

    # 创建评估器
    evaluator = ModelEvaluator(model, device)

    # 评估训练集
    print("\n评估训练集表现...")
    train_predictions, train_actuals, train_metrics = evaluator.evaluate(train_loader)

    # 评估验证集
    print("\n评估验证集表现...")
    val_predictions, val_actuals, val_metrics = evaluator.evaluate(val_loader)

    # 评估测试集
    print("\n评估测试集表现...")
    test_predictions, test_actuals, test_metrics = evaluator.evaluate(test_loader)

    # 计算额外的测试集指标
    test_pred_np = np.array(test_predictions).flatten()
    test_actual_np = np.array(test_actuals).flatten()

    # 计算相关系数R
    corr_coef = np.corrcoef(test_pred_np, test_actual_np)[0, 1]

    # 创建完整评估结果汇总表格
    evaluation_summary = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics,
        'test_additional': {
            'correlation_coefficient': corr_coef
        },
        'hyperparams': {
            'dropout_rate': optimal_dropout,
            'learning_rate': optimal_lr,
            'batch_size': optimal_batch_size
        }
    }

    # 保存完整评估结果
    save_experiment_results(
        evaluation_summary,
        Config.TEST_OUTPUT_DIR,
        f"{Config.DATASET_NAME}_evaluation_summary"
    )

    # 创建CSV文件保存所有数据集的指标
    metrics_df = pd.DataFrame({
        'Dataset': ['训练集', '验证集', '测试集'],
        'R²': [train_metrics['R²'], val_metrics['R²'], test_metrics['R²']],
        'RMSE': [train_metrics['RMSE'], val_metrics['RMSE'], test_metrics['RMSE']],
        'MAE': [train_metrics['MAE'], val_metrics['MAE'], test_metrics['MAE']],
        'R (相关系数)': [np.corrcoef(np.array(train_predictions).flatten(), np.array(train_actuals).flatten())[0, 1],
                         np.corrcoef(np.array(val_predictions).flatten(), np.array(val_actuals).flatten())[0, 1],
                         corr_coef]
    })

    metrics_csv_path = os.path.join(Config.TEST_OUTPUT_DIR, f"{Config.DATASET_NAME}_all_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False, encoding='utf-8-sig')
    print(f"所有数据集指标汇总已保存至: {metrics_csv_path}")

    # 保存测试结果详情
    test_results = {
        'predictions': test_predictions,
        'actuals': test_actuals,
        'metrics': test_metrics,
        'correlation_coefficient': corr_coef,
        'hyperparams': {
            'dropout_rate': optimal_dropout,
            'learning_rate': optimal_lr,
            'batch_size': optimal_batch_size
        }
    }
    save_experiment_results(
        test_results,
        Config.TEST_OUTPUT_DIR,
        f"{Config.DATASET_NAME}_test"
    )

    # 可视化测试结果
    test_visualizer = ResultVisualizer(
        dataset_name=Config.DATASET_NAME,
        output_dir=Config.TEST_OUTPUT_DIR
    )
    test_visualizer.visualize_all(None, None, test_predictions, test_actuals)

    # 额外创建测试集真实值与预测值的散点图
    create_prediction_scatter_plot(
        test_predictions,
        test_actuals,
        test_metrics,
        corr_coef,
        Config.TEST_OUTPUT_DIR,
        Config.DATASET_NAME
    )

    # 创建真实值与预测值的线图比较
    create_line_comparison_plot(
        test_predictions,
        test_actuals,
        Config.TEST_OUTPUT_DIR,
        Config.DATASET_NAME
    )

    print("\n=== 所有流程执行完毕 ===")

    # 打印最终评估结果
    print("\n=== 最终评估结果汇总 ===")
    print("\n训练集评估:")
    for metric_name, metric_value in train_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    print("\n验证集评估:")
    for metric_name, metric_value in val_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

    print("\n测试集评估:")
    for metric_name, metric_value in test_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    print(f"  相关系数 (R): {corr_coef:.4f}")


def create_prediction_scatter_plot(predictions, actuals, metrics, corr_coef, output_dir, dataset_name):
    plt.figure(figsize=(10, 8))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # 对角线从(0,0)到(1,1)
    plt.xlim(0, 1)  # 设置x轴范围
    plt.ylim(0, 1)  # 设置y轴范围
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(
        f'{dataset_name} - 预测值 vs 实际值\nR²: {metrics["R²"]:.4f}, RMSE: {metrics["RMSE"]:.4f}, R: {corr_coef:.4f}')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_scatter_plot.png'))
    plt.close()


def create_line_comparison_plot(predictions, actuals, output_dir, dataset_name):
    plt.figure(figsize=(15, 6))
    plt.plot(actuals, label='实际值', alpha=0.7)
    plt.plot(predictions, label='预测值', alpha=0.7)
    plt.ylim(0, 1)  # 设置y轴范围
    plt.xlabel('样本索引')
    plt.ylabel('值')
    plt.title(f'{dataset_name} - 实际值 vs 预测值对比')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_line_comparison.png'))
    plt.close()


if __name__ == "__main__":
    main()
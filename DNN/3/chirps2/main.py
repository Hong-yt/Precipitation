import os
import torch
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
import pandas as pd
import datetime

from config import Config
from dataset import PrecipitationDataProcessor, PrecipitationDataset
from model import PrecipitationDNN
from trainer import PrecipitationModelTrainer
from evaluator import ModelEvaluator
from utils import (set_seed, get_device, save_experiment_results, create_cv_folds,
                   save_cv_results, run_hyperparameter_search)
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

    features_df, target_df = data_processor.load_data()

    # 数据预处理（包括离群值处理、特征选择和目标变换）
    X, y = data_processor.preprocess_data(from_config=True, config=Config)

    # 如果需要进行超参数优化
    if Config.HYPERPARAMETER_TUNING:
        print("\n=== 开始超参数优化 ===")

        # 合并常规参数和特征工程参数
        all_params = Config.GRID_PARAMS.copy()

        # 只有在启用特征工程时才加入特征工程参数
        if Config.FEATURE_ENGINEERING:
            for key, value in Config.FEATURE_ENGINEERING_PARAMS.items():
                all_params[key] = value

        best_params, _ = run_hyperparameter_search(
            X, y,
            param_grid=all_params,
            config=Config,
            evaluator=ModelEvaluator,
            base_model_cls=PrecipitationDNN,
            data_processor=data_processor,
            max_combinations=Config.MAX_GRID_COMBINATIONS,
            cv_folds=Config.CV_FOLDS,
            verbose=True
        )
        print(f"\n应用最佳参数: {best_params}")

        # 将最佳参数应用到Config
        Config.update_params(best_params)

    # 创建交叉验证的数据分割
    folds = create_cv_folds(
        X, y,
        n_splits=Config.CV_FOLDS,
        shuffle=Config.CV_SHUFFLE,
        random_state=Config.SEED
    )

    # 存储所有折叠的预测结果和实际值
    all_predictions = []
    all_actuals = []
    all_indices = []  # 存储样本索引
    all_is_augmented = []  # 存储是否为增强样本的标志

    # 存储所有折叠的评估指标
    cv_metrics = {
        'R²': [],
        'RMSE': [],
        'MAE': []
    }

    # 存储每折的详细指标
    fold_metrics = []

    # 进行K折交叉验证
    for fold_idx, (train_index, test_index) in enumerate(folds, 1):
        print(f"\n=== 开始第 {fold_idx} 折交叉验证 ===")

        # 分割数据
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 保存当前折的标准化器
        fold_scaler_path = os.path.join(Config.CV_OUTPUT_DIR, f"{Config.DATASET_NAME}_fold{fold_idx}_scaler.pth")
        torch.save(scaler, fold_scaler_path)

        # 创建数据集和数据加载器
        train_dataset = PrecipitationDataset(X_train, y_train)
        test_dataset = PrecipitationDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            drop_last=False  # 修改为False，确保不丢弃任何样本
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            drop_last=False  # 修改为False，确保不丢弃任何样本
        )

        # 使用静态方法创建当前折的模型
        model = PrecipitationDNN.create_model_for_fold(
            input_size=X_train.shape[1],
            hidden_sizes=Config.HIDDEN_SIZES,
            dropout_rate=Config.DROPOUT_RATE,
            fold_idx=fold_idx,
            activation=Config.ACTIVATION
        )
        model.to(device)

        # 初始化训练器
        trainer = PrecipitationModelTrainer(
            model=model,
            dataset_name=f"{Config.DATASET_NAME}_fold{fold_idx}",
            device=device,
            optimizer_name=Config.OPTIMIZER,
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )

        # 训练模型
        print(f"=== 开始训练模型（第 {fold_idx} 折）===")
        train_losses, val_losses, _ = trainer.train(
            train_loader=train_loader,
            val_loader=test_loader,  # 使用测试集作为验证集
            epochs=Config.EPOCHS,
            patience=Config.PATIENCE
        )

        # 保存训练结果
        train_results = {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        save_experiment_results(
            train_results,
            Config.CV_OUTPUT_DIR,
            f"{Config.DATASET_NAME}_fold{fold_idx}_train"
        )

        # 测试模型
        print(f"=== 评估模型（第 {fold_idx} 折）===")
        evaluator = ModelEvaluator(model, device)
        predictions, actuals, metrics = evaluator.evaluate(test_loader)

        # 验证预测结果的数量
        print(f"第 {fold_idx} 折测试集样本数: {len(test_index)}")
        print(f"第 {fold_idx} 折预测结果数量: {len(predictions)}")
        assert len(predictions) == len(
            test_index), f"预测结果数量 ({len(predictions)}) 与测试集样本数 ({len(test_index)}) 不匹配"

        # 如果应用了目标变换，需要逆变换
        if data_processor.target_transformer is not None:
            print("应用目标变量的逆变换...")
            predictions = data_processor.inverse_transform_target(predictions)
            actuals = data_processor.inverse_transform_target(actuals)

            # 重新计算指标
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            metrics = {
                'R²': r2_score(actuals, predictions),
                'RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
                'MAE': mean_absolute_error(actuals, predictions),
                '平均相对误差(%)': np.mean(np.abs((predictions - actuals) / actuals * 100)),
                '最大相对误差(%)': np.max(np.abs((predictions - actuals) / actuals * 100)),
                '最小相对误差(%)': np.min(np.abs((predictions - actuals) / actuals * 100))
            }
            print("\n=== 逆变换后的评估指标 ===")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

        # 获取样本是否为增强样本的标志
        is_augmented = data_processor.get_augmentation_flags(test_index)

        # 存储当前折叠的预测结果和评估指标
        all_predictions.extend(predictions)
        all_actuals.extend(actuals)
        all_indices.extend(test_index)  # 保存测试集的索引
        all_is_augmented.extend(is_augmented)  # 保存是否为增强样本的标志
        fold_metrics.append(metrics)

        for key in cv_metrics.keys():
            cv_metrics[key].append(metrics[key])

        # 保存当前折叠的测试结果
        test_results = {
            'predictions': predictions,
            'actuals': actuals,
            'metrics': metrics,
            'indices': test_index.tolist(),  # 保存索引信息
            'is_augmented': is_augmented.tolist()  # 保存是否为增强样本的标志
        }
        save_experiment_results(
            test_results,
            Config.CV_OUTPUT_DIR,
            f"{Config.DATASET_NAME}_fold{fold_idx}_test"
        )

        # 创建并保存预测结果表格
        results_df = pd.DataFrame({
            '样本索引': test_index,
            '是否增强样本': is_augmented,
            '实际值': actuals,
            '预测值': predictions,
            '误差': predictions - actuals,
            '相对误差(%)': ((predictions - actuals) / actuals * 100)
        })
        results_df.to_csv(os.path.join(Config.CV_OUTPUT_DIR, f"{Config.DATASET_NAME}_fold{fold_idx}_results.csv"),
                          index=False)

    # 保存所有折叠的结果
    save_cv_results(
        all_predictions, all_actuals,
        cv_metrics, fold_metrics,
        Config.CV_OUTPUT_DIR,
        Config.DATASET_NAME
    )

    # 输出平均评估指标
    print("\n=== 交叉验证平均指标 ===")
    for key in cv_metrics:
        mean_val = np.mean(cv_metrics[key])
        std_val = np.std(cv_metrics[key])
        print(f"{key}: {mean_val:.4f} ± {std_val:.4f}")

    # 创建并保存所有折叠的汇总结果表格
    all_results_df = pd.DataFrame({
        '样本索引': all_indices,
        '是否增强样本': all_is_augmented,
        '实际值': all_actuals,
        '预测值': all_predictions,
        '误差': np.array(all_predictions) - np.array(all_actuals),
        '相对误差(%)': ((np.array(all_predictions) - np.array(all_actuals)) / np.array(all_actuals) * 100)
    })

    # 按样本索引排序
    all_results_df = all_results_df.sort_values('样本索引')

    # 分离原始样本和增强样本
    original_samples_df = all_results_df[~all_results_df['是否增强样本']].copy()
    augmented_samples_df = all_results_df[all_results_df['是否增强样本']].copy()

    # 删除"是否增强样本"列，因为已经分开保存
    original_samples_df = original_samples_df.drop('是否增强样本', axis=1)
    augmented_samples_df = augmented_samples_df.drop('是否增强样本', axis=1)

    # 保存到不同的CSV文件
    original_samples_df.to_csv(
        os.path.join(Config.CV_OUTPUT_DIR, f"{Config.DATASET_NAME}_original_samples_results.csv"), index=False)
    augmented_samples_df.to_csv(
        os.path.join(Config.CV_OUTPUT_DIR, f"{Config.DATASET_NAME}_augmented_samples_results.csv"), index=False)

    # 打印结果统计信息
    print("\n=== 预测结果统计信息 ===")
    print(f"总样本数: {len(all_results_df)}")
    print(f"原始样本数: {len(original_samples_df)}")
    print(f"增强样本数: {len(augmented_samples_df)}")

    # 验证原始样本数量是否为900
    expected_original_samples = 900
    if len(original_samples_df) != expected_original_samples:
        print(f"\n警告：原始样本数量 ({len(original_samples_df)}) 与预期数量 ({expected_original_samples}) 不符！")
        print("请检查数据加载和预测过程是否有样本丢失。")

    print("\n=== 原始样本统计信息 ===")
    print(f"平均预测值: {original_samples_df['预测值'].mean():.4f}")
    print(f"平均实际值: {original_samples_df['实际值'].mean():.4f}")
    print(f"平均误差: {original_samples_df['误差'].mean():.4f}")
    print(f"平均相对误差: {original_samples_df['相对误差(%)'].mean():.4f}%")

    print("\n=== 增强样本统计信息 ===")
    print(f"平均预测值: {augmented_samples_df['预测值'].mean():.4f}")
    print(f"平均实际值: {augmented_samples_df['实际值'].mean():.4f}")
    print(f"平均误差: {augmented_samples_df['误差'].mean():.4f}")
    print(f"平均相对误差: {augmented_samples_df['相对误差(%)'].mean():.4f}%")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"程序运行出错: {e}")
        traceback.print_exc()
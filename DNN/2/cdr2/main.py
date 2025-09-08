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
from visualizer import ResultVisualizer
from shap_analyzer import ShapAnalyzer, aggregate_shap_values
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

    # 存储所有折叠的评估指标
    cv_metrics = {
        'R²': [],
        'RMSE': [],
        'MAE': []
    }

    # 存储每折的详细指标
    fold_metrics = []

    # 存储每折的SHAP值和特征重要性
    all_fold_shap_values = []
    all_fold_shap_features = []
    all_fold_importance_dfs = []
    feature_names = data_processor.get_feature_names()

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
            drop_last=Config.DROP_LAST
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            drop_last=Config.DROP_LAST
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

        # 可视化训练结果
        train_visualizer = ResultVisualizer(
            dataset_name=f"{Config.DATASET_NAME}_fold{fold_idx}",
            output_dir=Config.CV_OUTPUT_DIR
        )
        train_visualizer.visualize_losses(train_losses, val_losses)

        # 测试模型
        print(f"=== 评估模型（第 {fold_idx} 折）===")
        evaluator = ModelEvaluator(model, device)
        predictions, actuals, metrics = evaluator.evaluate(test_loader)

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

        # 存储当前折叠的预测结果和评估指标
        all_predictions.extend(predictions)
        all_actuals.extend(actuals)
        fold_metrics.append(metrics)

        for key in cv_metrics.keys():
            cv_metrics[key].append(metrics[key])

        # 保存当前折叠的测试结果
        test_results = {
            'predictions': predictions,
            'actuals': actuals,
            'metrics': metrics
        }
        save_experiment_results(
            test_results,
            Config.CV_OUTPUT_DIR,
            f"{Config.DATASET_NAME}_fold{fold_idx}_test"
        )

        # 可视化预测结果
        test_visualizer = ResultVisualizer(
            dataset_name=f"{Config.DATASET_NAME}_fold{fold_idx}",
            output_dir=Config.CV_OUTPUT_DIR
        )
        test_visualizer.visualize_predictions(predictions, actuals)

        # ========== 添加SHAP分析 ==========
        if Config.SHAP_ANALYSIS:
            print(f"\n=== 开始SHAP分析（第 {fold_idx} 折）===")

            try:
                # 创建SHAP分析器
                shap_analyzer = ShapAnalyzer(model, device, feature_names=feature_names)
                shap_analyzer.set_fold_index(fold_idx)

                # 创建背景数据集
                # 使用一小部分训练数据作为背景数据
                background_size = min(Config.SHAP_BACKGROUND_SAMPLES, len(X_train))
                # 随机选择样本
                background_indices = np.random.choice(len(X_train), background_size, replace=False)
                background_data = X_train[background_indices]

                print(f"创建背景数据，大小: {background_data.shape}")

                # 创建SHAP解释器，使用TabularExplainer
                try:
                    explainer = shap_analyzer.use_tabular_explainer(background_data)

                    # 为了效率，只使用测试集的一部分样本进行SHAP分析
                    max_test_samples = min(400, len(X_test))  # 增加样本数量从100到400
                    if len(X_test) > max_test_samples:
                        print(f"为提高效率，仅选择{max_test_samples}个测试样本进行SHAP分析")
                        test_indices = np.random.choice(len(X_test), max_test_samples, replace=False)
                        X_test_subset = X_test[test_indices]
                    else:
                        X_test_subset = X_test

                    # 计算测试集的SHAP值
                    shap_values = shap_analyzer.compute_shap_values(X_test_subset)

                    # 检查SHAP值是否有效
                    if shap_values is not None and len(shap_values) > 0 and not np.isnan(shap_values).all():
                        print(f"成功计算SHAP值，形状: {shap_values.shape}")

                        # 保存当前折的SHAP值和特征重要性
                        importance_df = shap_analyzer.save_shap_values(
                            Config.SHAP_OUTPUT_DIR,
                            f"{Config.DATASET_NAME}_fold{fold_idx}",
                            X_test_subset,
                            fold_idx
                        )

                        # 仅当importance_df不为None时才添加到列表
                        if importance_df is not None:
                            # 保存SHAP值和特征用于后续汇总分析
                            all_fold_shap_values.append(shap_values)

                            # 将标准化的特征数据转换回原始空间，以便汇总分析时使用原始特征值
                            X_test_original = data_processor.inverse_transform_features(X_test_subset, scaler)
                            print(f"特征数据反向变换完成，形状: {X_test_original.shape}")
                            all_fold_shap_features.append(X_test_original)

                            all_fold_importance_dfs.append(importance_df)

                            # 可视化SHAP结果
                            try:
                                shap_visualizer = ResultVisualizer(
                                    dataset_name=f"{Config.DATASET_NAME}_fold{fold_idx}",
                                    output_dir=Config.SHAP_OUTPUT_DIR
                                )

                                # 特征重要性
                                shap_visualizer.visualize_shap_feature_importance(
                                    importance_df, top_n=Config.SHAP_TOP_FEATURES
                                )

                                # 创建特征分组的SHAP分析 - 这部分将显示SHAP依赖图
                                try:
                                    # 使用原始特征值进行SHAP依赖图可视化
                                    shap_visualizer.visualize_shap_feature_groups(
                                        shap_values, X_test_original, feature_names
                                    )

                                    # 注释掉不需要的三张图的生成代码
                                    # 新增：使用标准化特征值生成SHAP蜂群图
                                    # shap_visualizer.visualize_shap_beeswarm(
                                    #     shap_values, X_test_subset, feature_names
                                    # )

                                    # 新增：生成组合图（条形图+蜂群图）
                                    shap_visualizer.visualize_shap_combined(
                                        shap_values, X_test_original, feature_names, max_display=None  # 使用全部特征
                                    )
                                except Exception as e:
                                    print(f"生成SHAP依赖图或蜂群图时出错: {e}")
                                    import traceback
                                    traceback.print_exc()

                                print(f"第 {fold_idx} 折SHAP分析可视化完成")
                            except Exception as viz_error:
                                print(f"SHAP可视化时出错: {viz_error}")
                                import traceback
                                traceback.print_exc()
                        else:
                            print(f"第 {fold_idx} 折未能生成有效的特征重要性，跳过可视化")
                    else:
                        print(f"第 {fold_idx} 折计算的SHAP值无效，跳过可视化")
                except Exception as e:
                    print(f"SHAP分析时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    print("跳过当前折的SHAP分析，继续下一折...")
            except Exception as e:
                print(f"SHAP分析时出错: {e}")
                import traceback
                traceback.print_exc()
                print("跳过当前折的SHAP分析，继续下一折...")

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

    # 收集所有折的损失数据
    fold_losses = {}
    for fold_idx in range(1, Config.CV_FOLDS + 1):
        # 从保存的JSON文件中加载训练结果
        results_path = os.path.join(Config.CV_OUTPUT_DIR, f"{Config.DATASET_NAME}_fold{fold_idx}_train.json")
        if os.path.exists(results_path):
            with open(results_path, 'r', encoding='utf-8') as f:
                fold_result = json.load(f)
                fold_losses[fold_idx] = {
                    'train_losses': fold_result.get('train_losses', []),
                    'val_losses': fold_result.get('val_losses', [])
                }

    # 创建整体结果可视化
    overall_visualizer = ResultVisualizer(
        dataset_name=Config.DATASET_NAME,
        output_dir=Config.CV_OUTPUT_DIR
    )
    overall_visualizer.visualize_predictions(all_predictions, all_actuals)

    # 可视化所有折的最终损失和精度对比
    if fold_losses:
        overall_visualizer.visualize_cv_loss_and_accuracy(fold_losses, fold_metrics)

    overall_visualizer.visualize_feature_importance(data_processor, fold_metrics)

    # ========== 添加SHAP汇总分析 ==========
    if Config.SHAP_ANALYSIS:
        print("\n=== 开始SHAP汇总分析 ===")

        try:
            # 检查是否有足够的SHAP值进行汇总分析
            if all_fold_shap_values and len(all_fold_shap_values) > 0:
                print(f"收集到 {len(all_fold_shap_values)} 折的SHAP值")

                # 检查SHAP值是否包含NaN
                has_nan = False
                for shap_vals in all_fold_shap_values:
                    if np.isnan(shap_vals).any():
                        print(f"警告: SHAP值中包含NaN，将替换为0")
                        has_nan = True
                        break

                # 如果有NaN，清理所有数组
                if has_nan:
                    all_fold_shap_values = [np.nan_to_num(sv) for sv in all_fold_shap_values]
                    all_fold_shap_features = [np.nan_to_num(sf) for sf in all_fold_shap_features]

                # 使用aggregate_shap_values函数代替直接的np.vstack操作
                all_shap_values, all_features = aggregate_shap_values(all_fold_shap_values, all_fold_shap_features)

                if all_shap_values is not None and all_features is not None:
                    print(f"汇总SHAP值形状: {all_shap_values.shape}")
                    print(f"汇总特征形状: {all_features.shape}")

                    # 创建汇总可视化
                    try:
                        shap_overall_visualizer = ResultVisualizer(
                            dataset_name=f"{Config.DATASET_NAME}_overall",
                            output_dir=Config.SHAP_OUTPUT_DIR
                        )

                        # 如果至少有一个折有特征重要性数据
                        if all_fold_importance_dfs and len(all_fold_importance_dfs) > 0:
                            # 各折SHAP特征重要性比较
                            # 创建折叠索引到SHAP值的字典
                            fold_shap_dict = {}
                            for i, fold_idx in enumerate(range(1, len(all_fold_shap_values) + 1)):
                                fold_shap_dict[fold_idx] = all_fold_shap_values[i]

                            shap_overall_visualizer.visualize_shap_fold_comparison(
                                fold_shap_dict, all_features, feature_names,
                                max_display=Config.SHAP_TOP_FEATURES
                            )

                            # 生成汇总SHAP蜂群图
                            try:
                                # 注释掉不需要的蜂群图
                                # shap_overall_visualizer.visualize_shap_beeswarm(
                                #     all_shap_values, all_features, feature_names, max_display=30
                                # )
                                # print("汇总SHAP蜂群图生成完成")

                                # 新增：生成汇总组合图（条形图+蜂群图），显示所有特征
                                shap_overall_visualizer.visualize_shap_combined(
                                    all_shap_values, all_features, feature_names, max_display=None
                                )
                                print("汇总SHAP组合图生成完成")

                                # 新增：生成汇总特征依赖图
                                try:
                                    shap_overall_visualizer.visualize_shap_feature_groups(
                                        all_shap_values, all_features, feature_names
                                    )
                                    print("汇总SHAP依赖图生成完成")
                                except Exception as e:
                                    print(f"生成汇总SHAP依赖图时出错: {e}")
                                    import traceback
                                    traceback.print_exc()

                                # 保存汇总的SHAP分析结果
                                try:
                                    # 保存汇总的SHAP值
                                    shap_file = os.path.join(Config.SHAP_OUTPUT_DIR,
                                                             f"{Config.DATASET_NAME}_overall_shap_values.npy")
                                    np.save(shap_file, all_shap_values)
                                    print(f"汇总SHAP值已保存到 {shap_file}")

                                    # 保存汇总的特征数据
                                    features_file = os.path.join(Config.SHAP_OUTPUT_DIR,
                                                                 f"{Config.DATASET_NAME}_overall_features.npy")
                                    np.save(features_file, all_features)
                                    print(f"汇总特征数据已保存到 {features_file}")

                                    # 计算并保存汇总的特征重要性
                                    feature_importance = np.abs(all_shap_values).mean(axis=0)
                                    importance_df = pd.DataFrame({
                                        'Feature': feature_names,
                                        'Importance': feature_importance
                                    })
                                    importance_df = importance_df.sort_values('Importance', ascending=False)

                                    # 保存到CSV
                                    importance_file = os.path.join(Config.SHAP_OUTPUT_DIR,
                                                                   f"{Config.DATASET_NAME}_overall_importance.csv")
                                    importance_df.to_csv(importance_file, index=False)
                                    print(f"汇总特征重要性已保存到 {importance_file}")

                                    # 保存元数据
                                    metadata = {
                                        'dataset_name': f"{Config.DATASET_NAME}_overall",
                                        'shap_values_shape': all_shap_values.shape,
                                        'features_shape': all_features.shape,
                                        'fold_count': len(all_fold_shap_values),
                                        'feature_count': len(feature_names),
                                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    }

                                    # 保存元数据到JSON
                                    metadata_file = os.path.join(Config.SHAP_OUTPUT_DIR,
                                                                 f"{Config.DATASET_NAME}_overall_metadata.json")
                                    with open(metadata_file, 'w') as f:
                                        json.dump(metadata, f, indent=2)
                                    print(f"汇总元数据已保存到 {metadata_file}")

                                except Exception as e:
                                    print(f"保存汇总SHAP分析结果时出错: {e}")
                                    import traceback
                                    traceback.print_exc()

                                print("\n=== SHAP分析完成 ===")
                            except Exception as e:
                                print(f"生成汇总SHAP图时出错: {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            print("没有可用的SHAP值进行汇总分析")
                    except Exception as viz_error:
                        print(f"SHAP汇总可视化时出错: {viz_error}")
                        import traceback
                        traceback.print_exc()
            else:
                print("没有可用的SHAP值进行汇总分析")
        except Exception as e:
            print(f"SHAP汇总分析时出错: {e}")
            import traceback
            traceback.print_exc()
            print("跳过SHAP汇总分析")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"程序运行出错: {e}")
        traceback.print_exc()
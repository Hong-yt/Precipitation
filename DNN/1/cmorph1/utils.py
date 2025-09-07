import torch
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import json
import pandas as pd
from sklearn.model_selection import KFold
import itertools
import random
from collections import defaultdict
import time
import matplotlib.pyplot as plt
from visualizer import ResultVisualizer


def set_seed(torch_seed=42, numpy_seed=42):
    """设置随机种子以确保结果可重复"""
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    np.random.seed(numpy_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(torch_seed)


def get_device():
    """获取可用的计算设备（CPU或GPU）"""
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
        print(f"当前GPU内存使用情况: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    return device


def load_checkpoint(model, checkpoint_path, device):
    """加载模型检查点"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功加载检查点: {checkpoint_path}")
    else:
        print(f"未找到检查点文件: {checkpoint_path}")
    return model


def convert_to_serializable(obj):
    """将对象转换为可JSON序列化的格式"""
    if isinstance(obj, np.ndarray):
        return obj.astype(float).tolist()
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def save_experiment_results(results, output_dir, filename):
    """保存实验结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 如果结果是字典类型，将其转换为可序列化的格式
    if isinstance(results, dict):
        # 使用递归函数转换所有值
        serializable_results = convert_to_serializable(results)

        # 保存为JSON文件（使用UTF-8编码）
        json_path = os.path.join(output_dir, f"{filename}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=4)
        print(f"结果已保存到: {json_path}")

        # 如果包含预测结果，也保存为CSV
        if 'predictions' in results and 'actuals' in results:
            df = pd.DataFrame({
                '预测值': np.array(results['predictions']).astype(float),
                '实际值': np.array(results['actuals']).astype(float)
            })
            csv_path = os.path.join(output_dir, f"{filename}.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"预测结果已保存到: {csv_path}")
    else:
        # 如果不是字典类型，直接保存为文本文件（使用UTF-8编码）
        txt_path = os.path.join(output_dir, f"{filename}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(str(results))
        print(f"结果已保存到: {txt_path}")


def create_cv_folds(X, y, n_splits=5, shuffle=True, random_state=42):
    """
    创建交叉验证的数据分割

    参数:
    X: 特征数据
    y: 目标数据
    n_splits: 折数
    shuffle: 是否打乱数据
    random_state: 随机种子

    返回:
    folds: 包含(train_idx, test_idx)元组的列表
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    folds = list(kf.split(X))

    print(f"\n=== 创建{n_splits}折交叉验证数据分割 ===")
    for i, (train_idx, test_idx) in enumerate(folds):
        print(f"第{i + 1}折: 训练集大小 {len(train_idx)}, 测试集大小 {len(test_idx)}")

    return folds


def save_cv_results(all_predictions, all_actuals, cv_metrics, fold_metrics, output_dir, dataset_name):
    """
    保存交叉验证的所有结果

    参数:
    all_predictions: 所有折的预测值集合
    all_actuals: 所有折的实际值集合
    cv_metrics: 每折的评估指标
    fold_metrics: 每折详细的指标
    output_dir: 输出目录
    dataset_name: 数据集名称
    """
    os.makedirs(output_dir, exist_ok=True)

    # 确保预测值和实际值是NumPy数组
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)

    # 保存所有预测结果和实际值
    results_df = pd.DataFrame({
        '实际值': all_actuals,
        '预测值': all_predictions,
        '绝对误差': np.abs(all_predictions - all_actuals),
        '相对误差(%)': np.abs((all_predictions - all_actuals) / all_actuals * 100)
    })
    results_path = os.path.join(output_dir, f"{dataset_name}_cv_predictions.csv")
    results_df.to_csv(results_path, index=False, encoding='utf-8')

    # 保存每折的评估指标
    cv_df = pd.DataFrame(cv_metrics)
    cv_df.index = [f"折{i + 1}" for i in range(len(cv_metrics['R²']))]
    cv_df.loc['平均值'] = cv_df.mean()
    cv_df.loc['标准差'] = cv_df.std()
    cv_path = os.path.join(output_dir, f"{dataset_name}_cv_metrics.csv")
    cv_df.to_csv(cv_path, encoding='utf-8')

    # 保存为JSON格式
    save_experiment_results(
        {
            'predictions': all_predictions,
            'actuals': all_actuals,
            'cv_metrics': cv_metrics,
            'fold_metrics': fold_metrics
        },
        output_dir,
        f"{dataset_name}_cv_results"
    )

    print(f"交叉验证结果已保存到: {output_dir}")
    return results_df, cv_df


def generate_parameter_grid(param_grid, max_combinations=20):
    """
    生成参数网格，如果参数组合太多则随机采样

    参数:
    param_grid: 参数网格字典
    max_combinations: 最大组合数

    返回:
    param_list: 参数组合列表
    """
    # 计算所有可能的组合数
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    # 计算可能的组合总数
    total_combinations = 1
    for v in values:
        total_combinations *= len(v)

    print(f"参数网格中有 {total_combinations} 种可能的组合")

    # 如果组合数量不多，直接返回全部组合
    if total_combinations <= max_combinations:
        print(f"使用全部 {total_combinations} 种组合")
        param_list = []
        for combo in itertools.product(*values):
            param_dict = {keys[i]: combo[i] for i in range(len(keys))}
            param_list.append(param_dict)
        return param_list

    # 否则随机采样
    print(f"组合太多，随机采样 {max_combinations} 种组合")
    random.seed(42)  # 设置随机种子以确保可重复性

    param_list = []
    for _ in range(max_combinations):
        param_dict = {}
        for i, key in enumerate(keys):
            # 随机选择一个值
            param_dict[key] = random.choice(values[i])

        # 确保没有重复的组合
        if param_dict not in param_list:
            param_list.append(param_dict)

    # 如果随机采样导致重复，可能最终组合数少于max_combinations
    print(f"最终生成了 {len(param_list)} 种组合")
    return param_list


def run_hyperparameter_search(X, y, param_grid, config, evaluator, base_model_cls, data_processor,
                              max_combinations=20, cv_folds=5, verbose=True, torch=None):
    """
    运行超参数搜索

    参数:
    X: 特征数据
    y: 目标数据
    param_grid: 参数网格
    config: 配置对象
    evaluator: 评估器类
    base_model_cls: 基础模型类
    data_processor: 数据处理器对象
    max_combinations: 最大组合数
    cv_folds: 交叉验证折数
    verbose: 是否打印详细信息

    返回:
    best_params: 最佳参数组合
    results: 所有参数组合的评估结果
    """
    print(f"\n=== 开始超参数搜索 ===")
    start_time = time.time()

    # 创建交叉验证分割
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=config.SEED)
    splits = list(kf.split(X))

    # 生成参数组合
    param_combinations = generate_parameter_grid(param_grid, max_combinations)

    # 记录每组参数的评估结果
    results = []

    # 获取设备
    try:
        # 确保torch正确导入
        import torch
        device = get_device()
        print(f"使用设备: {device}")
    except Exception as e:
        print(f"获取设备时出错: {e}")
        print("使用CPU作为备选设备")
        # 如果发生错误，使用CPU作为备选
        import torch
        device = torch.device("cpu")

    # 保存最佳参数和性能
    best_score = -float('inf')
    best_params = None
    best_metrics = None

    # 遍历每种参数组合
    for i, params in enumerate(param_combinations):
        print(f"\n参数组合 {i + 1}/{len(param_combinations)}: {params}")

        # 更新配置
        for k, v in params.items():
            setattr(config, k.upper(), v)

        # 初始化评估指标
        fold_scores = []
        fold_metrics = []

        # 进行交叉验证
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            if verbose:
                print(f"\n折 {fold_idx + 1}/{cv_folds}")

            # 划分数据
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # 标准化
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # 准备数据加载器
            from torch.utils.data import DataLoader, TensorDataset
            import torch

            train_dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32)
            )
            test_dataset = TensorDataset(
                torch.tensor(X_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32)
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=params.get('batch_size', config.BATCH_SIZE),
                shuffle=True,
                drop_last=config.DROP_LAST
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=params.get('batch_size', config.BATCH_SIZE),
                drop_last=config.DROP_LAST
            )

            # 创建模型
            model = base_model_cls(
                input_size=X_train.shape[1],
                hidden_sizes=params.get('hidden_sizes', config.HIDDEN_SIZES),
                dropout_rate=params.get('dropout_rate', config.DROPOUT_RATE),
                activation=params.get('activation', config.ACTIVATION)
            )
            model.to(device)

            # 初始化训练器
            from trainer import PrecipitationModelTrainer
            trainer = PrecipitationModelTrainer(
                model=model,
                dataset_name=f"hyperparam_search_fold{fold_idx}",
                device=device,
                optimizer_name=params.get('optimizer', config.OPTIMIZER),
                lr=params.get('learning_rate', config.LEARNING_RATE),
                weight_decay=params.get('weight_decay', config.WEIGHT_DECAY)
            )

            # 训练模型
            train_losses, val_losses, _ = trainer.train(
                train_loader=train_loader,
                val_loader=test_loader,
                epochs=config.EPOCHS,
                patience=params.get('patience', config.PATIENCE),
                log_interval=20  # 减少日志输出频率
            )

            # 评估模型
            with torch.no_grad():
                model.eval()
                test_preds = []
                test_targets = []

                for features, targets in test_loader:
                    features, targets = features.to(device), targets.to(device)
                    outputs = model(features)
                    test_preds.extend(outputs.cpu().numpy())
                    test_targets.extend(targets.cpu().numpy())

                test_preds = np.array(test_preds).flatten()
                test_targets = np.array(test_targets).flatten()

                # 如果应用了对数变换，需要逆变换
                if hasattr(data_processor, 'target_transformer') and data_processor.target_transformer is not None:
                    test_preds = data_processor.inverse_transform_target(test_preds)
                    test_targets = data_processor.inverse_transform_target(test_targets)

                # 计算R²
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                r2 = r2_score(test_targets, test_preds)
                rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
                mae = mean_absolute_error(test_targets, test_preds)

                metrics = {
                    'R²': r2,
                    'RMSE': rmse,
                    'MAE': mae
                }

                fold_scores.append(r2)
                fold_metrics.append(metrics)

                # 创建每折的输出目录
                fold_output_dir = os.path.join(config.HYPERPARAM_OUTPUT_DIR,
                                               f"params_combo_{i + 1}/fold_{fold_idx + 1}")
                os.makedirs(fold_output_dir, exist_ok=True)

                # 保存测试结果
                test_results = {
                    'predictions': test_preds.tolist(),
                    'actuals': test_targets.tolist(),
                    'metrics': metrics
                }
                save_experiment_results(
                    test_results,
                    fold_output_dir,
                    f"{config.DATASET_NAME}_params_combo_{i + 1}_fold_{fold_idx + 1}_test"
                )

                # 保存训练损失和验证损失
                train_results = {
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }
                save_experiment_results(
                    train_results,
                    fold_output_dir,
                    f"{config.DATASET_NAME}_params_combo_{i + 1}_fold_{fold_idx + 1}_train"
                )

                if verbose:
                    print(f"折 {fold_idx + 1} R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")

        # 计算平均性能
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        print(f"参数组合 {i + 1} 平均 R² = {mean_score:.4f} ± {std_score:.4f}")

        # 记录结果
        result = {
            'params': params,
            'mean_r2': mean_score,
            'std_r2': std_score,
            'fold_scores': fold_scores,
            'fold_metrics': fold_metrics
        }
        results.append(result)

        # 为当前参数组合创建输出目录
        param_output_dir = os.path.join(config.HYPERPARAM_OUTPUT_DIR, f"params_combo_{i + 1}")
        os.makedirs(param_output_dir, exist_ok=True)

        # 保存当前参数组合的结果
        combo_result = {
            'params': params,
            'mean_r2': mean_score,
            'std_r2': std_score,
            'fold_metrics': fold_metrics,
            'fold_scores': fold_scores
        }
        save_experiment_results(combo_result, param_output_dir, f"{config.DATASET_NAME}_params_combo_{i + 1}")

        # 保存每折的详细指标
        cv_metrics = {
            'R²': [metrics['R²'] for metrics in fold_metrics],
            'RMSE': [metrics['RMSE'] for metrics in fold_metrics],
            'MAE': [metrics['MAE'] for metrics in fold_metrics]
        }

        # 创建一个表格展示当前参数组合
        param_df = pd.DataFrame([params])
        param_df.to_csv(os.path.join(param_output_dir, f"{config.DATASET_NAME}_params.csv"), index=False)

        # 创建一个表格展示每折的性能指标
        fold_df = pd.DataFrame(cv_metrics)
        fold_df.index = [f"折{i + 1}" for i in range(len(cv_metrics['R²']))]
        fold_df.loc['平均值'] = fold_df.mean()
        fold_df.loc['标准差'] = fold_df.std()
        fold_df.to_csv(os.path.join(param_output_dir, f"{config.DATASET_NAME}_fold_metrics.csv"), encoding='utf-8')

        # 创建当前参数组合的可视化结果
        param_visualizer = ResultVisualizer(
            dataset_name=f"{config.DATASET_NAME}_params_combo_{i + 1}",
            output_dir=param_output_dir
        )

        # 可视化当前参数组合的交叉验证结果
        param_visualizer.visualize_cv_metrics(cv_metrics)

        print(f"参数组合 {i + 1} 的结果已保存到: {param_output_dir}")

        # 更新最佳参数
        if mean_score > best_score:
            best_score = mean_score
            best_params = params.copy()
            best_metrics = {
                'mean_r2': mean_score,
                'std_r2': std_score,
                'fold_metrics': fold_metrics
            }
            print(f"✓ 发现新的最佳参数组合! 平均 R² = {mean_score:.4f}")

    # 计算搜索耗时
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n超参数搜索完成，耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    print(f"最佳参数组合: {best_params}")
    print(f"最佳性能: R² = {best_metrics['mean_r2']:.4f} ± {best_metrics['std_r2']:.4f}")

    # 保存搜索结果
    save_hyperparameter_search_results(results, best_params, best_metrics, config.HYPERPARAM_OUTPUT_DIR,
                                       config.DATASET_NAME)

    # 将最佳参数设置回配置
    for k, v in best_params.items():
        setattr(config, k.upper(), v)

    return best_params, results


def save_hyperparameter_search_results(results, best_params, best_metrics, output_dir, dataset_name):
    """
    保存超参数搜索结果

    参数:
    results: 搜索结果
    best_params: 最佳参数组合
    best_metrics: 最佳性能指标
    output_dir: 输出目录
    dataset_name: 数据集名称
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存所有结果
    all_results = {
        'best_params': best_params,
        'best_metrics': best_metrics,
        'all_results': results
    }

    # 保存为JSON
    save_experiment_results(all_results, output_dir, f"{dataset_name}_hyperparameter_search")

    # 创建结果DataFrame
    results_df = []
    for result in results:
        row = result['params'].copy()
        row['mean_r2'] = result['mean_r2']
        row['std_r2'] = result['std_r2']
        results_df.append(row)

    results_df = pd.DataFrame(results_df)

    # 按R²降序排序
    results_df = results_df.sort_values('mean_r2', ascending=False)

    # 保存为CSV
    csv_path = os.path.join(output_dir, f"{dataset_name}_hyperparameter_search_results.csv")
    results_df.to_csv(csv_path, index=False)

    # 创建汇总图表，展示所有参数组合的性能比较
    summary_visualizer = ResultVisualizer(
        dataset_name=f"{dataset_name}_hyperparameter_search_summary",
        output_dir=output_dir
    )

    # 创建一个条形图，显示所有参数组合的R²值
    plt.figure(figsize=(12, 8))
    param_indices = list(range(1, len(results) + 1))
    mean_r2_values = [result['mean_r2'] for result in results]
    std_r2_values = [result['std_r2'] for result in results]

    # 按照R²值排序
    sorted_indices = np.argsort(mean_r2_values)[::-1]
    sorted_params = [param_indices[i] for i in sorted_indices]
    sorted_mean_r2 = [mean_r2_values[i] for i in sorted_indices]
    sorted_std_r2 = [std_r2_values[i] for i in sorted_indices]

    plt.bar(sorted_params, sorted_mean_r2, yerr=sorted_std_r2, capsize=5, alpha=0.7)
    plt.xlabel('参数组合编号')
    plt.ylabel('平均 R²')
    plt.title('所有参数组合性能对比')
    plt.xticks(sorted_params)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 添加数值标签
    for i, (val, err) in enumerate(zip(sorted_mean_r2, sorted_std_r2)):
        plt.text(sorted_params[i], val + 0.01, f'{val:.4f}±{err:.4f}',
                 ha='center', va='bottom', rotation=90, fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_param_combos_comparison.png"), dpi=300)
    plt.close()

    # 为每个超参数绘制其对R²的影响
    plot_hyperparameter_effects(results, output_dir, dataset_name)

    print(f"超参数搜索结果已保存到: {output_dir}")
    return results_df


def plot_hyperparameter_effects(results, output_dir, dataset_name):
    """
    绘制每个超参数对模型性能的影响

    参数:
    results: 搜索结果
    output_dir: 输出目录
    dataset_name: 数据集名称
    """
    # 提取所有不同的参数
    param_keys = results[0]['params'].keys()

    # 为每个参数创建一个图表
    for param in param_keys:
        # 收集参数值和对应的R²
        param_values = []
        r2_values = []

        for result in results:
            param_values.append(str(result['params'][param]))  # 转换为字符串以处理列表等
            r2_values.append(result['mean_r2'])

        # 计算每个参数值的平均R²
        param_r2_dict = defaultdict(list)
        for param_val, r2 in zip(param_values, r2_values):
            param_r2_dict[param_val].append(r2)

        avg_r2 = {param_val: np.mean(r2s) for param_val, r2s in param_r2_dict.items()}
        std_r2 = {param_val: np.std(r2s) for param_val, r2s in param_r2_dict.items()}

        # 按参数值排序
        sorted_params = sorted(avg_r2.keys(), key=lambda x: (len(x), x))

        # 创建图表
        plt.figure(figsize=(10, 6))

        # 参数值
        x = list(range(len(sorted_params)))
        # 平均R²
        y = [avg_r2[p] for p in sorted_params]
        # 标准差
        yerr = [std_r2[p] for p in sorted_params]

        plt.bar(x, y, yerr=yerr, capsize=5, alpha=0.7)
        plt.xticks(x, sorted_params, rotation=45 if len(max(sorted_params, key=len)) > 10 else 0)
        plt.xlabel(param)
        plt.ylabel('平均 R²')
        plt.title(f'{param} 对模型性能的影响')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 添加数值标签
        for i, (val, err) in enumerate(zip(y, yerr)):
            plt.text(i, val + 0.01, f'{val:.4f}±{err:.4f}',
                     ha='center', va='bottom', rotation=0, fontsize=9)

        plt.tight_layout()

        # 保存图表
        save_path = os.path.join(output_dir, f"{dataset_name}_param_effect_{param}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    print(f"参数效果图已保存到: {output_dir}")
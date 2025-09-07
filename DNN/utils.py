import torch
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import json
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import time
from dataset import PrecipitationDataset
from trainer import PrecipitationModelTrainer


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def cross_validation_on_train_set(X_train_val, y_train_val, model_class, n_splits=5,
                                  batch_size=64, epochs=100, device='cuda',
                                  hidden_sizes=[256, 128, 96, 64, 32], dropout_rate=0.25,
                                  lr=0.0005, weight_decay=5e-5, patience=25):
    """
    在训练集上执行K折交叉验证，包括超参数搜索

    参数:
    X_train_val: 用于训练和验证的特征数据
    y_train_val: 用于训练和验证的目标变量
    model_class: 模型类
    n_splits: 交叉验证折数
    batch_size: 默认批量大小（当使用batch_sizes参数时，此参数会被忽略）
    epochs: 训练轮数
    device: 计算设备
    hidden_sizes: 隐藏层大小
    dropout_rate: 默认dropout率（当使用dropout_rates参数时，此参数会被忽略）
    lr: 默认学习率（当使用learning_rates参数时，此参数会被忽略）
    weight_decay: L2正则化系数
    patience: 早停耐心值

    返回:
    包含各折结果和最佳超参数的字典
    """
    print(f"\n=== 开始{n_splits}折交叉验证 (包括超参数搜索) ===")
    print(f"训练+验证数据集大小: {X_train_val.shape}")

    # 从config.py导入超参数配置
    from config import Config

    # 超参数搜索空间
    batch_sizes = Config.BATCH_SIZES if hasattr(Config, 'BATCH_SIZES') else [batch_size]
    dropout_rates = Config.DROPOUT_RATES if hasattr(Config, 'DROPOUT_RATES') else [dropout_rate]
    learning_rates = Config.LEARNING_RATES if hasattr(Config, 'LEARNING_RATES') else [lr]

    print(f"\n超参数搜索空间:")
    print(f"  批量大小: {batch_sizes}")
    print(f"  Dropout率: {dropout_rates}")
    print(f"  学习率: {learning_rates}")

    # 初始化K折交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 存储所有超参数组合的结果
    all_results = []

    # 遍历所有超参数组合
    for bs in batch_sizes:
        for dr in dropout_rates:
            for lr in learning_rates:
                print(f"\n=== 评估超参数组合: batch_size={bs}, dropout_rate={dr}, learning_rate={lr} ===")

                # 用于存储每一折的指标
                fold_metrics = {
                    'R2': [],
                    'RMSE': [],
                    'MAE': [],
                    'R': [],
                    'MSE': []
                }

                # 对每一折进行训练和评估
                for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
                    print(f"\n=== 训练第 {fold + 1}/{n_splits} 折 ===")

                    # 拆分数据
                    X_train_fold, X_val_fold = X_train_val[train_idx], X_train_val[val_idx]
                    y_train_fold, y_val_fold = y_train_val[train_idx], y_train_val[val_idx]

                    # 标准化特征
                    scaler = StandardScaler()
                    X_train_fold = scaler.fit_transform(X_train_fold)
                    X_val_fold = scaler.transform(X_val_fold)

                    # 创建数据集和数据加载器
                    train_dataset = PrecipitationDataset(X_train_fold, y_train_fold)
                    val_dataset = PrecipitationDataset(X_val_fold, y_val_fold)

                    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=bs)

                    # 初始化模型
                    input_size = X_train_fold.shape[1]
                    model = model_class(input_size=input_size, hidden_sizes=hidden_sizes, dropout_rate=dr)
                    model.to(device)

                    # 训练模型
                    trainer = PrecipitationModelTrainer(model, f"fold_{fold + 1}", device, learning_rate=lr,
                                                        weight_decay=weight_decay)
                    train_losses, val_losses = trainer.train(
                        train_loader=train_loader,
                        val_loader=val_loader,
                        epochs=epochs
                    )

                    # 评估模型
                    model.eval()
                    with torch.no_grad():
                        # 预测验证集
                        val_preds = []
                        val_targets = []

                        for features, targets in val_loader:
                            features, targets = features.to(device), targets.to(device)
                            outputs = model(features)
                            val_preds.extend(outputs.cpu().numpy())
                            val_targets.extend(targets.cpu().numpy())

                        # 计算指标
                        val_preds = np.array(val_preds).flatten()
                        val_targets = np.array(val_targets).flatten()

                        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                        r2 = r2_score(val_targets, val_preds)
                        rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
                        mae = mean_absolute_error(val_targets, val_preds)
                        mse = mean_squared_error(val_targets, val_preds)
                        r = np.corrcoef(val_preds, val_targets)[0, 1]

                        # 存储结果
                        fold_metrics['R2'].append(r2)
                        fold_metrics['RMSE'].append(rmse)
                        fold_metrics['MAE'].append(mae)
                        fold_metrics['MSE'].append(mse)
                        fold_metrics['R'].append(r)

                        print(f"第 {fold + 1} 折验证集评估结果:")
                        print(f"  R²: {r2:.4f}")
                        print(f"  RMSE: {rmse:.4f}")
                        print(f"  MAE: {mae:.4f}")
                        print(f"  R: {r:.4f}")

                # 计算当前超参数组合的平均指标
                mean_metrics = {}
                std_metrics = {}
                for metric in fold_metrics:
                    mean_metrics[metric] = np.mean(fold_metrics[metric])
                    std_metrics[metric] = np.std(fold_metrics[metric])

                # 保存当前超参数组合的结果
                current_result = {
                    'batch_size': bs,
                    'dropout_rate': dr,
                    'learning_rate': lr,
                    'mean_metrics': mean_metrics,
                    'std_metrics': std_metrics,
                    'fold_metrics': fold_metrics
                }
                all_results.append(current_result)

                print(f"\n=== 超参数组合评估结果: batch_size={bs}, dropout_rate={dr}, learning_rate={lr} ===")
                for metric, value in mean_metrics.items():
                    std = std_metrics[metric]
                    print(f"{metric}: {value:.4f} ± {std:.4f}")

    # 找出最佳超参数组合
    best_result = max(all_results, key=lambda x: x['mean_metrics']['R2'])
    best_batch_size = best_result['batch_size']
    best_dropout_rate = best_result['dropout_rate']
    best_learning_rate = best_result['learning_rate']
    best_r2 = best_result['mean_metrics']['R2']

    print("\n=== 交叉验证超参数搜索结果汇总 ===")
    print(f"最佳超参数组合:")
    print(f"  Batch Size: {best_batch_size}")
    print(f"  Dropout Rate: {best_dropout_rate}")
    print(f"  Learning Rate: {best_learning_rate}")
    print(f"  验证集平均R²: {best_r2:.4f}")

    return {
        'results': all_results,
        'best_result': best_result,
        'best_batch_size': best_batch_size,
        'best_dropout': best_dropout_rate,
        'best_lr': best_learning_rate
    }
import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import shap
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import json
import datetime
import traceback
from config import Config


# 添加聚合SHAP值的函数
def aggregate_shap_values(all_fold_shap_values, all_fold_shap_features=None, fold_weights=None):
    """
    聚合多个折叠的SHAP值

    参数:
    all_fold_shap_values: 所有折叠的SHAP值列表
    all_fold_shap_features: 所有折叠的特征值列表 (可选)
    fold_weights: 每个折叠的权重，如果为None则等权重 (可选)

    返回:
    aggregated_shap_values: 聚合后的SHAP值
    aggregated_features: 聚合后的特征 (如果提供了all_fold_shap_features)
    """
    if not all_fold_shap_values or len(all_fold_shap_values) == 0:
        print("没有SHAP值可聚合")
        return None, None

    print(f"聚合 {len(all_fold_shap_values)} 折的SHAP值...")

    # 预处理：检查NaN并替换为0
    for i in range(len(all_fold_shap_values)):
        if np.isnan(all_fold_shap_values[i]).any():
            print(f"警告: 第 {i + 1} 折的SHAP值包含NaN，将替换为0")
            all_fold_shap_values[i] = np.nan_to_num(all_fold_shap_values[i])

            if all_fold_shap_features is not None and i < len(all_fold_shap_features):
                all_fold_shap_features[i] = np.nan_to_num(all_fold_shap_features[i])

    # 如果没有提供权重，则所有折叠等权重
    if fold_weights is None:
        fold_weights = [1.0] * len(all_fold_shap_values)
    else:
        # 确保权重长度与折叠数一致
        if len(fold_weights) != len(all_fold_shap_values):
            print(f"警告: 权重数量 ({len(fold_weights)}) 与折叠数 ({len(all_fold_shap_values)}) 不符，使用等权重")
            fold_weights = [1.0] * len(all_fold_shap_values)

        # 归一化权重
        weight_sum = sum(fold_weights)
        fold_weights = [w / weight_sum for w in fold_weights]

    try:
        # 简单地垂直堆叠所有折叠的SHAP值
        aggregated_shap_values = np.vstack(all_fold_shap_values)

        if all_fold_shap_features is not None:
            # 如果提供了特征，也垂直堆叠它们
            aggregated_features = np.vstack(all_fold_shap_features)
            print(f"聚合完成! SHAP值形状: {aggregated_shap_values.shape}, 特征形状: {aggregated_features.shape}")
            return aggregated_shap_values, aggregated_features
        else:
            print(f"聚合完成! SHAP值形状: {aggregated_shap_values.shape}")
            return aggregated_shap_values, None

    except Exception as e:
        print(f"聚合SHAP值时出错: {e}")
        traceback.print_exc()
        return None, None


class ShapAnalyzer:
    def __init__(self, model, device, feature_names=None):
        """
        初始化SHAP分析器

        参数:
        model: 训练好的模型
        device: 运行设备 (CPU/GPU)
        feature_names: 特征名称列表
        """
        self.model = model
        self.device = device
        self.feature_names = feature_names
        self.model.to(device)
        self.model.eval()
        self.explainer = None
        self.shap_values = None
        self.fold_idx = None

    def set_fold_index(self, fold_idx):
        """设置当前折叠索引"""
        self.fold_idx = fold_idx

    def use_tabular_explainer(self, background_data, batch_size=None):
        """
        使用TabularExplainer创建SHAP解释器
        (实际上调用create_explainer方法，保留此方法是为了兼容性)

        参数:
        background_data: 用于建立解释器期望值的背景数据
        batch_size: 批次大小

        返回:
        explainer: 创建的SHAP解释器
        """
        print("使用TabularExplainer创建SHAP解释器（实际使用KernelExplainer）...")
        return self.create_explainer(background_data, batch_size)

    def create_explainer(self, background_data, batch_size=None):
        """
        创建SHAP解释器，使用KernelExplainer替代DeepExplainer

        参数:
        background_data: 用于建立解释器期望值的背景数据
        batch_size: 批次大小（仅为兼容性保留，不使用）
        """
        print("转换为使用KernelExplainer...")

        # 确保数据格式正确
        if isinstance(background_data, torch.Tensor):
            background_data = background_data.cpu().numpy()

        # 为KernelExplainer定义一个预测函数
        def f(x):
            # 确保模型处于评估模式
            self.model.eval()

            with torch.no_grad():
                # 转换为torch张量
                if not isinstance(x, torch.Tensor):
                    x = torch.FloatTensor(x).to(self.device)
                # 确保形状正确
                if len(x.shape) == 1:
                    x = x.unsqueeze(0)

                # 批处理大小至少为2，解决BatchNorm层的问题
                if x.shape[0] == 1:
                    # 对单个样本，复制一份以创建批次大小为2
                    x_batch = torch.cat([x, x], dim=0)
                    # 前向传播
                    out = self.model(x_batch)
                    # 只返回第一个样本的输出
                    return out[0:1].cpu().numpy()
                else:
                    # 前向传播
                    out = self.model(x)
                    # 返回为numpy数组
                    return out.cpu().numpy()

        # 限制背景数据大小以提高效率
        max_background_size = Config.SHAP_BACKGROUND_SAMPLES  # 使用配置中的SHAP_BACKGROUND_SAMPLES
        if len(background_data) > max_background_size:
            print(f"限制背景数据大小为{max_background_size}来提高效率")
            # 随机采样子集
            indices = np.random.choice(len(background_data), max_background_size, replace=False)
            background_data = background_data[indices]

        print(f"使用背景数据，形状: {background_data.shape}")

        try:
            print("创建KernelExplainer...")
            self.explainer = shap.KernelExplainer(f, background_data)
            print("SHAP解释器创建完成")
        except Exception as e:
            print(f"创建SHAP解释器时出错: {e}")
            raise ValueError("无法创建SHAP解释器，请尝试不同的模型或SHAP方法")

        return self.explainer

    def compute_shap_values(self, X, batch_size=None, nsamples=None):
        """
        计算测试集样本的SHAP值。
        如果出错，将尝试单个样本进行计算。

        参数:
        X: 测试样本特征
        batch_size: 批次大小（目前不使用，因为KernelExplainer自行管理批处理）
        nsamples: 用于KernelExplainer的样本数量，如果未指定则使用Config.SHAP_NSAMPLES

        返回:
        SHAP值（numpy数组）
        """
        if self.explainer is None:
            raise ValueError("请先调用create_explainer方法创建解释器")

        print("开始计算SHAP值...")

        # 确保X是numpy数组
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        # 限制样本数量以加快速度
        max_samples = 400
        if len(X) > max_samples:
            print(f"样本数过多，限制为{max_samples}，将随机取样...")
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]

        print(f"计算{len(X)}个样本的SHAP值...")

        try:
            # 使用配置中的nsamples参数或默认值
            if nsamples is None:
                nsamples = getattr(Config, 'SHAP_NSAMPLES', 100)
            print(f"使用{nsamples}个样本进行SHAP估计")

            # 使用KernelExplainer计算SHAP值
            shap_values = self.explainer.shap_values(X, nsamples=nsamples)

            # 检查SHAP值是否合理
            if isinstance(shap_values, list) and len(shap_values) > 0:
                shap_values = shap_values[0]  # 取第一个输出

            if shap_values is None or len(shap_values) == 0:
                print("未能计算得到有效的SHAP值")
                return None

            print(f"成功计算SHAP值，形状: {shap_values.shape}")
            # 将SHAP值保存到实例变量中，这样save_shap_values可以使用
            self.shap_values = shap_values
            return shap_values

        except Exception as e:
            print(f"计算SHAP值时发生错误: {str(e)}")
            print("详细错误信息：")
            traceback.print_exc()

            print("尝试逐个样本计算SHAP值...")

            # 逐个样本计算SHAP值
            all_shap_values = []
            for i, sample in enumerate(tqdm(X)):
                try:
                    # 使用与上面相同的nsamples参数
                    sample_shap = self.explainer.shap_values(sample.reshape(1, -1), nsamples=nsamples)

                    if isinstance(sample_shap, list) and len(sample_shap) > 0:
                        sample_shap = sample_shap[0]  # 取第一个输出

                    all_shap_values.append(sample_shap)
                except Exception as sample_e:
                    print(f"样本{i}计算失败: {str(sample_e)}")
                    # 添加一个零向量代替失败的样本
                    zero_shape = (1, X.shape[1])
                    all_shap_values.append(np.zeros(zero_shape))

            if len(all_shap_values) > 0:
                # 将所有样本的SHAP值合并为一个数组
                combined_shap_values = np.vstack(all_shap_values)
                print(f"成功对{len(all_shap_values)}个样本单独计算SHAP值，形状: {combined_shap_values.shape}")
                # 将合并后的SHAP值保存到实例变量中
                self.shap_values = combined_shap_values
                return combined_shap_values
            else:
                print("未能计算得到有效的SHAP值")
                return None

    def _compute_shap_values_in_batches(self, data, batch_size=32):
        """
        按批次计算SHAP值

        参数:
        data: 要计算SHAP值的数据
        batch_size: 批处理大小

        返回:
        all_shap_values: 所有样本的SHAP值
        """
        print(f"按批次计算SHAP值，批大小: {batch_size}")
        all_shap_values = []

        # 计算批次数
        n_batches = int(np.ceil(len(data) / batch_size))

        # 使用tqdm显示进度
        for i in tqdm(range(n_batches), desc="批次计算SHAP值"):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(data))
            batch_data = data[start_idx:end_idx]

            try:
                # 检查explainer类型
                explainer_type = type(self.explainer).__name__

                if explainer_type == "Explainer" or "Tabular" in explainer_type:
                    # 新版SHAP API
                    batch_shap = self.explainer(batch_data)
                    if hasattr(batch_shap, "values"):
                        batch_shap = batch_shap.values
                else:
                    # 传统API
                    batch_shap = self.explainer.shap_values(batch_data)

                # 如果结果是列表（多输出模型），取第一个
                if isinstance(batch_shap, list):
                    batch_shap = batch_shap[0]

                all_shap_values.append(batch_shap)
            except Exception as e:
                print(f"计算批次 {i + 1}/{n_batches} 的SHAP值时出错: {e}")
                print("尝试逐个样本计算此批次...")

                # 对当前批次逐个样本计算
                batch_shap = self._compute_shap_values_one_by_one(batch_data)
                all_shap_values.append(batch_shap)

        # 合并所有批次的SHAP值
        try:
            combined_shap = np.vstack(all_shap_values)
            print(f"批次计算完成，SHAP值形状: {combined_shap.shape}")
            # 将SHAP值保存到实例变量中
            self.shap_values = combined_shap
            return combined_shap
        except Exception as e:
            print(f"合并批次SHAP值时出错: {e}")
            # 如果合并失败，返回零值
            zero_values = np.zeros((len(data), data.shape[1]))
            # 保存零值到实例变量
            self.shap_values = zero_values
            return zero_values

    def _compute_shap_values_one_by_one(self, data):
        """
        逐个样本计算SHAP值

        参数:
        data: 要计算SHAP值的数据

        返回:
        all_shap_values: 所有样本的SHAP值
        """
        print("逐个样本计算SHAP值...")
        all_shap_values = []
        explainer_type = type(self.explainer).__name__

        for i in tqdm(range(len(data)), desc="单样本计算SHAP值"):
            try:
                sample = data[i:i + 1]

                # 根据解释器类型选择调用方法
                if explainer_type == "Explainer" or "Tabular" in explainer_type:
                    # 新版SHAP API
                    sample_shap = self.explainer(sample)
                    if hasattr(sample_shap, "values"):
                        sample_shap = sample_shap.values
                else:
                    # 传统API
                    sample_shap = self.explainer.shap_values(sample)

                if isinstance(sample_shap, list):
                    sample_shap = sample_shap[0]

                all_shap_values.append(sample_shap)
            except Exception as e:
                print(f"计算样本 {i} 的SHAP值时出错: {e}")
                # 用零填充
                zero_shap = np.zeros((1, data.shape[1]))
                all_shap_values.append(zero_shap)

        # 合并所有样本的SHAP值
        try:
            combined_shap = np.vstack(all_shap_values)
            print(f"单样本计算完成，SHAP值形状: {combined_shap.shape}")
            # 将SHAP值保存到实例变量中
            self.shap_values = combined_shap
            return combined_shap
        except Exception as e:
            print(f"合并单样本SHAP值时出错: {e}")
            # 如果合并失败，返回零值
            zero_values = np.zeros((len(data), data.shape[1]))
            # 保存零值到实例变量
            self.shap_values = zero_values
            return zero_values

    def save_shap_values(self, output_dir, dataset_name, data=None, fold_idx=None):
        """
        保存SHAP值和相关分析结果，支持npy和CSV格式

        参数:
        output_dir: 输出目录
        dataset_name: 数据集名称
        data: 原始数据 (可选)
        fold_idx: 折叠索引 (可选)

        返回:
        importance_df: 特征重要性DataFrame，如果无法生成则返回None
        """
        if self.shap_values is None:
            print("没有SHAP值可保存")
            return None

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 准备文件名
        fold_str = f"_fold{fold_idx}" if fold_idx is not None else ""
        filename_base = f"{dataset_name}{fold_str}_shap"

        # 1. 保存SHAP值为npy格式
        shap_file_npy = os.path.join(output_dir, f"{filename_base}_values.npy")
        np.save(shap_file_npy, self.shap_values)
        print(f"SHAP值(npy)已保存到 {shap_file_npy}")

        # 2. 保存SHAP值为CSV格式（新增）
        shap_file_csv = os.path.join(output_dir, f"{filename_base}_values.csv")
        try:
            # 创建DataFrame
            if self.feature_names:
                shap_df = pd.DataFrame(self.shap_values, columns=self.feature_names)
            else:
                shap_df = pd.DataFrame(self.shap_values)

            # 添加样本ID列（新增）
            shap_df.insert(0, 'sample_id', np.arange(len(shap_df)))
            shap_df.to_csv(shap_file_csv, index=False)
            print(f"SHAP值(CSV)已保存到 {shap_file_csv}")
        except Exception as e:
            print(f"保存SHAP值为CSV时出错: {e}")

        # 保存特征重要性
        importance_df = None
        if self.feature_names is not None:
            # 计算每个特征的平均绝对SHAP值
            feature_importance = np.abs(self.shap_values).mean(axis=0)

            # 创建特征重要性数据框
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            })

            # 按重要性排序
            importance_df = importance_df.sort_values('importance', ascending=False)

            # 保存到CSV
            importance_file = os.path.join(output_dir, f"{filename_base}_importance.csv")
            importance_df.to_csv(importance_file, index=False)
            print(f"特征重要性已保存到 {importance_file}")

            # 保存重要性图表
            plt.figure(figsize=(10, 8))
            plt.barh(importance_df['feature'].values[:24], importance_df['importance'].values[:24])
            plt.xlabel('平均|SHAP值|')
            plt.title('SHAP特征重要性 (前24)')
            plt.tight_layout()

            # 保存图表
            plot_file = os.path.join(output_dir, f"{filename_base}_importance_plot.png")
            plt.savefig(plot_file, dpi=300)
            plt.close()
            print(f"特征重要性图表已保存到 {plot_file}")

        # 保存数据集信息
        if data is not None:
            # 如果数据太大，只保存一部分
            max_samples = min(1500, len(data))
            if len(data) > max_samples:
                # 随机抽样
                indices = np.random.choice(len(data), max_samples, replace=False)
                data_subset = data[indices]
            else:
                data_subset = data

            # 3. 保存数据子集为npy格式（原有）
            data_file_npy = os.path.join(output_dir, f"{filename_base}_data_subset.npy")
            np.save(data_file_npy, data_subset)
            print(f"数据子集(npy)已保存到 {data_file_npy}")

            # 4. 保存数据子集为CSV格式（新增）
            data_file_csv = os.path.join(output_dir, f"{filename_base}_data_subset.csv")
            try:
                # 创建DataFrame
                if self.feature_names:
                    data_df = pd.DataFrame(data_subset, columns=self.feature_names)
                else:
                    data_df = pd.DataFrame(data_subset)

                # 添加样本ID列（新增）
                data_df.insert(0, 'sample_id', np.arange(len(data_df)))
                data_df.to_csv(data_file_csv, index=False)
                print(f"数据子集(CSV)已保存到 {data_file_csv}")
            except Exception as e:
                print(f"保存数据子集为CSV时出错: {e}")

        # 保存元数据
        metadata = {
            'dataset_name': dataset_name,
            'fold_index': fold_idx,
            'shap_values_shape': self.shap_values.shape,
            'feature_count': len(self.feature_names) if self.feature_names is not None else None,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # 保存元数据到JSON
        with open(os.path.join(output_dir, f"{filename_base}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"SHAP分析结果已成功保存到 {output_dir}")

        return importance_df

    def get_top_features(self, n=10):
        """
        获取最重要的n个特征

        参数:
        n: 返回的特征数量

        返回:
        top_features: 最重要特征的列表
        top_indices: 最重要特征的索引
        """
        if self.shap_values is None:
            print("没有SHAP值可分析")
            return [], []

        if self.feature_names is None:
            print("没有特征名称可用")
            return [], []

        # 计算平均绝对SHAP值
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)

        # 获取索引排序（按重要性降序）
        top_indices = np.argsort(-mean_abs_shap)[:n]

        # 获取对应的特征名称
        top_features = [self.feature_names[i] for i in top_indices]

        return top_features, top_indices

    def get_feature_importance(self):
        """
        获取所有特征的重要性评分

        返回:
        importance_df: 包含特征名称和重要性的DataFrame，按重要性降序排列
        """
        if self.shap_values is None or self.feature_names is None:
            print("没有SHAP值或特征名称可用")
            return None

        # 计算平均绝对SHAP值
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)

        # 创建并排序DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        })
        importance_df = importance_df.sort_values('importance', ascending=False)

        return importance_df
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import warnings
import scipy.stats as stats

warnings.filterwarnings("ignore")


# 数据加载与预处理类
# 这个类负责处理数据的加载、预处理和划分，让代码结构更清晰
class PrecipitationDataProcessor:
    def __init__(self, dataset_name, features_path, target_path):
        """
        初始化数据处理器

        参数:
        dataset_name (str): 数据集名称（如'imerg', 'sm2rain'等）
        features_path (str): 特征数据文件路径，包含各种输入特征
        target_path (str): 目标数据（实际R值）文件路径，包含真实的相关系数
        """
        self.dataset_name = dataset_name
        self.features_path = features_path
        self.target_path = target_path
        self.scaler = None
        self.feature_selector = None
        self.target_transformer = None
        self.selected_features = None

    def validate_data_format(self, df, min_cols=1, max_cols=None, file_type=""):
        """增强型数据验证，确保所有值为数值类型"""
        actual_cols = df.shape[1]
        if actual_cols < min_cols:
            raise ValueError(f"{file_type}数据格式错误：至少需要{min_cols}列，实际{actual_cols}列")
        if max_cols is not None and actual_cols > max_cols:
            raise ValueError(f"{file_type}数据格式错误：最多允许{max_cols}列，实际{actual_cols}列")

        # 检查并转换数据类型
        non_numeric_cols = []
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_cols.append(col)
                try:
                    # 尝试转换为数值类型
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    raise ValueError(f"{file_type}列'{col}'包含无法转换的非数值数据: {e}")

        if non_numeric_cols:
            print(f"警告: {file_type}中{len(non_numeric_cols)}列包含非数值数据，已尝试转换")

        # 检查转换后是否有非数值类型
        if not all(df[col].dtype in [np.int64, np.float64] for col in df.columns):
            non_numeric_types = [col for col in df.columns if df[col].dtype not in [np.int64, np.float64]]
            raise ValueError(f"{file_type}转换后仍有非数值类型列: {non_numeric_types}")

        return df

    def load_data(self):
        print(f"\n=== 正在加载数据 ===")

        # 加载数据
        self.features_df = pd.read_excel(self.features_path).iloc[:, 2:]
        self.target_df = pd.read_excel(self.target_path).iloc[:, 2:]

        # 验证并转换数据类型
        self.features_df = self.validate_data_format(
            self.features_df, min_cols=1, file_type="特征")
        self.target_df = self.validate_data_format(
            self.target_df, min_cols=1, max_cols=1, file_type="目标")

        # 打印数据信息
        print("\n数据加载结果:")
        print(f"特征数据类型:\n{self.features_df.dtypes}")
        print(f"目标数据类型:\n{self.target_df.dtypes}")
        print(f"特征数据形状: {self.features_df.shape}")
        print(f"目标数据形状: {self.target_df.shape}")

        return self.features_df, self.target_df

    def remove_outliers(self, X, y, threshold=3.0):
        print("\n=== 移除离群值 ===")

        # 确保数据类型正确
        if not (X.dtype.kind in 'fiu' and y.dtype.kind in 'fiu'):
            raise TypeError("输入数据必须为数值类型")

        # 合并数据计算z-score
        data = np.hstack((X, y))
        n_samples = data.shape[0]
        mask = np.ones(n_samples, dtype=bool)

        for col_idx in range(data.shape[1]):
            col_data = data[:, col_idx]

            # 跳过常量列
            if np.all(col_data == col_data[0]):
                continue

            col_mean = np.nanmean(col_data)  # 忽略NaN值
            col_std = np.nanstd(col_data)

            if col_std == 0:
                continue

            col_z_scores = np.abs((col_data - col_mean) / col_std)
            col_mask = col_z_scores < threshold
            mask = mask & col_mask

        # 应用掩码
        X_clean = X[mask]
        y_clean = y[mask]

        removed_count = X.shape[0] - X_clean.shape[0]
        if X.shape[0] > 0:
            removed_percent = removed_count / X.shape[0] * 100
        else:
            removed_percent = 0

        print(f"移除了 {removed_count} 个离群样本 ({removed_percent:.2f}%)")
        print(f"剩余样本数: {X_clean.shape[0]}")

        return X_clean, y_clean

    def apply_log_transform(self, y, offset=0.01):
        """
        对目标变量应用对数变换

        参数:
        y: 目标变量
        offset: 小偏移量，避免log(0)问题

        返回:
        y_transformed: 变换后的目标变量
        """
        print("\n=== 对目标变量应用对数变换 ===")

        # 检查是否有负值
        if np.any(y < 0):
            min_val = np.min(y)
            offset = abs(min_val) + offset
            print(f"目标变量中存在负值，增加偏移量至 {offset}")

        # 应用对数变换
        y_transformed = np.log(y + offset)

        # 保存变换器参数，用于预测时的逆变换
        self.target_transformer = {'type': 'log', 'offset': offset}

        print(f"对数变换完成，偏移量: {offset}")

        return y_transformed

    def inverse_transform_target(self, y_pred):
        """
        将预测结果从变换空间转回原始空间

        参数:
        y_pred: 变换空间中的预测值

        返回:
        y_original: 原始空间中的预测值
        """
        if self.target_transformer is None:
            return y_pred

        if self.target_transformer['type'] == 'log':
            offset = self.target_transformer['offset']
            return np.exp(y_pred) - offset

        return y_pred

    def select_features(self, X, y, threshold=0.01):
        """
        使用随机森林进行特征选择

        参数:
        X: 特征矩阵
        y: 目标变量
        threshold: 特征重要性阈值

        返回:
        X_selected: 选择后的特征矩阵
        """
        print("\n=== 进行特征选择 ===")

        # 使用随机森林模型评估特征重要性
        rf = RandomForestRegressor(n_estimators=100, random_state=42)

        # 训练模型
        rf.fit(X, y.ravel())

        # 获取特征重要性
        importances = rf.feature_importances_

        # 找出重要性大于阈值的特征索引
        important_indices = np.where(importances > threshold)[0]

        if len(important_indices) == 0:
            print(f"没有特征重要性大于阈值 {threshold}，使用所有特征")
            self.selected_features = list(range(X.shape[1]))
            return X

        # 保存所选特征的索引，用于后续新数据
        self.selected_features = important_indices.tolist()

        # 应用特征选择
        X_selected = X[:, important_indices]

        print(f"特征选择完成，从 {X.shape[1]} 个特征中选出 {X_selected.shape[1]} 个")
        print(f"被选中的特征索引: {self.selected_features}")

        return X_selected

    def create_interaction_features(self, X, degree=2, interaction_only=False):
        """
        创建特征之间的交互项

        参数:
        X: 特征矩阵
        degree: 多项式次数
        interaction_only: 是否只包含交互项而不包含自身的幂

        返回:
        X_interaction: 包含交互特征的矩阵
        """
        print("\n=== 创建交互特征 ===")

        # 确保X是numpy数组并且是浮点类型
        if not isinstance(X, np.ndarray):
            print(f"输入X不是numpy数组，类型为: {type(X)}，尝试转换...")
            try:
                X = np.array(X, dtype=np.float64)
            except Exception as e:
                print(f"转换为numpy数组失败: {e}")
                return X

        try:
            # 创建多项式特征生成器
            poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)

            # 应用变换
            X_interaction = poly.fit_transform(X)

            # 保存特征变换器，用于后续新数据
            self.poly_transformer = poly

            print(f"交互特征创建完成，特征数从 {X.shape[1]} 增加到 {X_interaction.shape[1]}")

            return X_interaction
        except Exception as e:
            print(f"创建交互特征时出错: {e}")
            print("返回原始特征矩阵，不创建交互特征")
            return X

    def create_statistical_features(self, X):
        """
        为每个样本创建统计特征，如均值、标准差、分位数等

        参数:
        X: 特征矩阵

        返回:
        X_with_stats: 包含统计特征的矩阵
        """
        print("\n=== 创建统计特征 ===")

        # 确保X是numpy数组并且是浮点类型
        if not isinstance(X, np.ndarray):
            print(f"输入X不是numpy数组，类型为: {type(X)}，尝试转换...")
            try:
                X = np.array(X, dtype=np.float64)
            except Exception as e:
                print(f"转换为numpy数组失败: {e}")
                return X

        try:
            # 对每个样本计算统计量
            row_mean = np.mean(X, axis=1).reshape(-1, 1)
            row_std = np.std(X, axis=1).reshape(-1, 1)
            row_min = np.min(X, axis=1).reshape(-1, 1)
            row_max = np.max(X, axis=1).reshape(-1, 1)
            row_median = np.median(X, axis=1).reshape(-1, 1)
            row_q25 = np.percentile(X, 25, axis=1).reshape(-1, 1)
            row_q75 = np.percentile(X, 75, axis=1).reshape(-1, 1)
            row_range = row_max - row_min
            row_iqr = row_q75 - row_q25

            # 组合所有特征
            X_with_stats = np.hstack((
                X,
                row_mean, row_std, row_min, row_max, row_median,
                row_q25, row_q75, row_range, row_iqr
            ))

            print(f"统计特征创建完成，特征数从 {X.shape[1]} 增加到 {X_with_stats.shape[1]}")

            return X_with_stats
        except Exception as e:
            print(f"创建统计特征时出错: {e}")
            print("返回原始特征矩阵，不创建统计特征")
            return X

    def apply_nonlinear_transformations(self, X):
        """
        应用多种非线性变换来捕捉复杂关系

        参数:
        X: 特征矩阵

        返回:
        X_transformed: 包含非线性变换特征的矩阵
        """
        print("\n=== 应用非线性变换 ===")

        # 确保X是numpy数组并且是浮点类型
        if not isinstance(X, np.ndarray):
            print(f"输入X不是numpy数组，类型为: {type(X)}，尝试转换...")
            try:
                X = np.array(X, dtype=np.float64)
            except Exception as e:
                print(f"转换为numpy数组失败: {e}")
                return X

        # 确保是浮点类型
        if X.dtype not in [np.float64, np.float32]:
            print(f"警告: X的数据类型为 {X.dtype}，转换为float64...")
            try:
                X = X.astype(np.float64)
            except Exception as e:
                print(f"转换为float64失败: {e}")
                return X

        try:
            # 创建基本变换，使用安全的方式
            X_sqrt = np.sqrt(np.abs(X))
            X_square = np.power(X, 2)  # 使用np.power代替X**2
            X_log = np.log1p(np.abs(X))  # 使用log1p(x) = log(1+x)

            # 组合所有变换
            X_transformed = np.hstack((X, X_sqrt, X_square, X_log))

            # 替换无穷和NaN值
            X_transformed = np.nan_to_num(X_transformed, nan=0, posinf=0, neginf=0)

            print(f"非线性变换完成，特征数从 {X.shape[1]} 增加到 {X_transformed.shape[1]}")

            return X_transformed

        except Exception as e:
            print(f"应用非线性变换时出错: {e}")
            print("返回原始特征矩阵，不进行非线性变换")
            return X

    def generate_time_lag_features(self, X, lag_indices, lag_steps=[1, 2, 3]):
        """
        生成时间滞后特征（适用于时序数据）

        参数:
        X: 特征矩阵
        lag_indices: 要创建滞后特征的列索引
        lag_steps: 滞后步长列表

        返回:
        X_with_lags: 包含滞后特征的矩阵
        """
        print("\n=== 生成滞后特征 ===")

        # 确保X是numpy数组
        if not isinstance(X, np.ndarray):
            print(f"输入X不是numpy数组，类型为: {type(X)}，尝试转换...")
            try:
                X = np.array(X, dtype=np.float64)
            except Exception as e:
                print(f"转换为numpy数组失败: {e}")
                return X

        try:
            # 将X转换为DataFrame以便于操作
            X_df = pd.DataFrame(X)
            lag_features = []

            # 为每个选定的特征和滞后步长创建滞后特征
            for col in lag_indices:
                for lag in lag_steps:
                    lag_name = f"{col}_lag_{lag}"
                    X_df[lag_name] = X_df[col].shift(lag)

            # 填充NaN值（滞后导致的前几行）
            X_df = X_df.fillna(0)

            # 转回numpy数组
            X_with_lags = X_df.values

            print(f"滞后特征生成完成，特征数从 {X.shape[1]} 增加到 {X_with_lags.shape[1]}")

            return X_with_lags
        except Exception as e:
            print(f"生成滞后特征时出错: {e}")
            print("返回原始特征矩阵，不创建滞后特征")
            return X

    def augment_data_with_noise(self, X, y, noise_factor=0.05, n_samples=None):
        """
        通过添加随机噪声来增强数据集

        参数:
        X: 特征矩阵
        y: 目标变量
        noise_factor: 噪声因子，控制噪声的大小
        n_samples: 生成的新样本数量，默认与原始样本数相同

        返回:
        X_aug: 增强后的特征矩阵
        y_aug: 增强后的目标变量
        """
        print("\n=== 数据增强 ===")

        # 确保X和y是numpy数组
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            print("输入不是numpy数组，尝试转换...")
            try:
                X = np.array(X, dtype=np.float64)
                y = np.array(y, dtype=np.float64)
            except Exception as e:
                print(f"转换为numpy数组失败: {e}")
                return X, y

        try:
            if n_samples is None:
                n_samples = X.shape[0]

            # 计算每个特征的标准差作为噪声幅度的基础
            X_std = np.std(X, axis=0)
            y_std = np.std(y, axis=0)

            # 生成随机噪声
            X_noise = np.random.normal(0, X_std * noise_factor, (n_samples, X.shape[1]))
            y_noise = np.random.normal(0, y_std * noise_factor, (n_samples, y.shape[1]))

            # 随机选择原始样本并添加噪声
            indices = np.random.choice(X.shape[0], n_samples, replace=True)
            X_sampled = X[indices]
            y_sampled = y[indices]

            # 添加噪声创建新样本
            X_noisy = X_sampled + X_noise
            y_noisy = y_sampled + y_noise

            # 合并原始数据和增强数据
            X_aug = np.vstack((X, X_noisy))
            y_aug = np.vstack((y, y_noisy))

            print(f"数据增强完成，样本数从 {X.shape[0]} 增加到 {X_aug.shape[0]}")

            return X_aug, y_aug
        except Exception as e:
            print(f"数据增强时出错: {e}")
            print("返回原始数据，不进行数据增强")
            return X, y

    def preprocess_data(self, from_config=True, config=None):
        print(f"\n=== 数据预处理 ===")

        # 删除包含缺失值的行
        original_size = self.features_df.shape[0]
        self.features_df = self.features_df.dropna()
        self.target_df = self.target_df.loc[self.features_df.index]

        # 转换为numpy数组
        X = self.features_df.values.astype(np.float64)  # 确保浮点类型
        y = self.target_df.values.reshape(-1, 1).astype(np.float64)

        # 保存特征名
        self.feature_names = list(self.features_df.columns)

        # 如果从配置读取处理参数
        if from_config and config is not None:
            if config.OUTLIER_REMOVAL:
                X, y = self.remove_outliers(X, y, config.OUTLIER_THRESHOLD)

            # 特征工程 - 应用各种特征变换和增强
            if hasattr(config, 'FEATURE_ENGINEERING') and config.FEATURE_ENGINEERING:
                # 创建非线性变换
                if hasattr(config, 'NONLINEAR_TRANSFORM') and config.NONLINEAR_TRANSFORM:
                    X = self.apply_nonlinear_transformations(X)

                # 创建交互特征
                if hasattr(config, 'INTERACTION_FEATURES') and config.INTERACTION_FEATURES:
                    interaction_degree = getattr(config, 'INTERACTION_DEGREE', 2)
                    X = self.create_interaction_features(X, degree=interaction_degree)

                # 创建统计特征
                if hasattr(config, 'STATISTICAL_FEATURES') and config.STATISTICAL_FEATURES:
                    X = self.create_statistical_features(X)

                # 创建时间滞后特征（如果适用）
                if hasattr(config, 'TIME_LAG_FEATURES') and config.TIME_LAG_FEATURES:
                    lag_cols = getattr(config, 'LAG_COLUMNS', list(range(min(5, X.shape[1]))))
                    lag_steps = getattr(config, 'LAG_STEPS', [1, 2, 3])
                    X = self.generate_time_lag_features(X, lag_cols, lag_steps)

            # 数据增强
            if hasattr(config, 'DATA_AUGMENTATION') and config.DATA_AUGMENTATION:
                noise_factor = getattr(config, 'NOISE_FACTOR', 0.05)
                aug_samples = getattr(config, 'AUG_SAMPLES', int(X.shape[0] * 0.5))
                X, y = self.augment_data_with_noise(X, y, noise_factor, aug_samples)

            # 特征选择（放在特征工程之后，可能会过滤掉一些工程化的特征）
            if config.FEATURE_SELECTION:
                X = self.select_features(X, y, config.FEATURE_IMPORTANCE_THRESHOLD)

            # 对目标变量进行对数变换
            if config.LOG_TRANSFORM:
                y = self.apply_log_transform(y, config.LOG_OFFSET)

        print(f"最终特征矩阵形状: {X.shape}, 目标变量形状: {y.shape}")

        return X, y

    def create_data_split_for_cv(self, X, y, n_splits=5, random_state=42):
        """
        为交叉验证准备数据分割索引

        参数:
        X: 特征矩阵
        y: 目标变量
        n_splits: 交叉验证折数
        random_state: 随机种子

        返回:
        splits: 包含(train_idx, test_idx)元组的列表
        """
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(kf.split(X))

        print(f"\n=== 准备{n_splits}折交叉验证 ===")
        print(f"总样本数: {len(X)}")
        for fold, (train_idx, test_idx) in enumerate(splits):
            print(f"第{fold + 1}折: 训练集大小 {len(train_idx)}, 测试集大小 {len(test_idx)}")

        return splits


# 自定义数据集类，继承自PyTorch的Dataset类
# 这个类将numpy数组转换为PyTorch可以处理的格式
class PrecipitationDataset(Dataset):
    def __init__(self, features, targets):
        """
        初始化数据集

        参数:
        features: 特征数据，numpy数组
        targets: 目标数据，numpy数组
        """

        print("初始化数据集...")

        # 将numpy数组转换为PyTorch张量，并指定数据类型为float32
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        print(f"数据集大小: {self.features.shape}, {self.targets.shape}")

    def __len__(self):
        """返回数据集中样本的数量"""
        return len(self.features)

    def __getitem__(self, idx):
        """根据索引返回一个数据样本"""
        return self.features[idx], self.targets[idx]
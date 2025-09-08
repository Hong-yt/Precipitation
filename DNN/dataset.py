import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
import warnings

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
        self.scaler = StandardScaler()
        self.poly = None  # 多项式特征转换器，将在preprocess_data中初始化

    def validate_data_format(self, df, min_cols=1, max_cols=None, file_type=""):
        """
        验证数据格式是否符合要求

        参数:
        df: DataFrame对象
        min_cols: 最小列数要求
        max_cols: 最大列数要求（可选）
        file_type: 文件类型（'features' 或 'target'）
        """
        actual_cols = df.shape[1]
        if actual_cols < min_cols:
            raise ValueError(f"{file_type}数据格式错误：至少需要{min_cols}列，实际{actual_cols}列")
        if max_cols is not None and actual_cols > max_cols:
            raise ValueError(f"{file_type}数据格式错误：最多允许{max_cols}列，实际{actual_cols}列")

        # 验证所有列是否为数值类型
        if not all(df[col].dtype in [np.int64, np.float64] for col in df.columns):
            raise ValueError(f"{file_type}数据格式错误：所有列必须为数值类型")

    def load_data(self):
        """
        加载特征数据和目标数据

        返回:
        features_df: 包含所有输入特征的DataFrame
        target_df: 包含目标变量(R值)的DataFrame
        """

        print(f"\n=== 正在加载数据 ===")

        # 加载特征数据，忽略前两列（行列号）
        # self.features_df = pd.read_csv(self.features_path).iloc[:, 2:]
        self.features_df = pd.read_excel(self.features_path).iloc[:, 2:]

        # 加载目标数据（实际站点的 R 值），忽略前两列（如果目标数据也有行列号）
        # self.target_df = pd.read_csv(self.target_path).iloc[:, 2:]
        self.target_df = pd.read_excel(self.target_path).iloc[:, 2:]

        # 打印数据形状，便于检查数据加载是否正确
        print("\n数据加载结果:")
        print(f"特征数据前5行:\n{self.features_df.head()}")
        print(f"目标数据前5行:\n{self.target_df.head()}")
        print(f"shape of feature: {self.features_df.shape}")  # 预期形状大约为 (8000, 20)
        print(f"shape of target: {self.target_df.shape}")  # 预期形状大约为 (8000, 1)

        return self.features_df, self.target_df

    def add_polynomial_features(self, X, degree=2):
        """
        添加多项式特征

        参数:
        X: 特征矩阵
        degree: 多项式的度

        返回:
        增强后的特征矩阵
        """
        # 强制限制多项式特征的阶数为2，防止特征爆炸
        degree = min(degree, 2)

        if self.poly is None:
            self.poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
            X_poly = self.poly.fit_transform(X)
        else:
            X_poly = self.poly.transform(X)

        print(f"多项式特征扩展: 原始特征数 {X.shape[1]}, 扩展后特征数 {X_poly.shape[1]}")
        return X_poly

    def add_interaction_features(self, X_df):
        """
        添加特征交互项

        参数:
        X_df: 特征DataFrame

        返回:
        添加交互特征后的DataFrame
        """
        # 选择数值型特征计算交互项
        columns = X_df.columns
        X_interaction = X_df.copy()

        # 为重要特征对创建交互项
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                # 创建乘积交互项
                col_name = f"{columns[i]}_x_{columns[j]}"
                X_interaction[col_name] = X_df[columns[i]] * X_df[columns[j]]

        print(f"交互特征添加: 原始特征数 {len(columns)}, 添加交互项后特征数 {X_interaction.shape[1]}")
        return X_interaction

    def feature_engineering(self, X, add_poly_features=True, add_interactions=True):
        """
        对数据应用特征工程技术，但不增强数据量

        参数:
        X: 特征矩阵（已经标准化过的）
        add_poly_features: 是否添加多项式特征
        add_interactions: 是否添加特征交互项

        返回:
        处理后的特征矩阵
        """
        # 如果X是numpy数组，转换为DataFrame以便添加交互特征
        if add_interactions and isinstance(X, np.ndarray):
            # 创建临时列名
            col_names = [f'feature_{i}' for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=col_names)

            # 添加交互特征
            X_df = self.add_interaction_features(X_df)

            # 转回numpy数组
            X = X_df.values

        # 添加多项式特征
        if add_poly_features:
            X = self.add_polynomial_features(X, degree=2)

        return X

    def augment_with_noise(self, X, y, noise_level=0.05, n_samples=None):
        """
        通过添加高斯噪声来增强数据

        参数:
        X: 特征矩阵
        y: 目标变量
        noise_level: 噪声水平（标准差的比例）
        n_samples: 要生成的新样本数量，默认为原始数据大小的50%

        返回:
        X_augmented, y_augmented: 增强后的特征和目标变量
        """
        if n_samples is None:
            n_samples = int(X.shape[0] * 0.5)  # 默认增加50%的样本

        # 随机选择样本进行增强
        indices = np.random.choice(X.shape[0], n_samples, replace=True)
        X_selected = X[indices]
        y_selected = y[indices]

        # 为每个特征添加不同的高斯噪声
        noise = np.random.normal(0, noise_level, X_selected.shape)
        std_per_feature = np.std(X, axis=0)
        noise = noise * std_per_feature  # 按特征标准差调整噪声幅度

        X_noisy = X_selected + noise
        # 为目标变量添加轻微噪声
        y_noisy = y_selected + np.random.normal(0, noise_level * 0.1, y_selected.shape)

        # 合并原始数据和增强数据
        X_augmented = np.vstack([X, X_noisy])
        y_augmented = np.vstack([y, y_noisy])

        print(f"数据增强: 原始样本数 {X.shape[0]}, 增强后样本数 {X_augmented.shape[0]}")
        return X_augmented, y_augmented

    def preprocess_data(self, add_poly_features=True, add_interactions=True, augment_data=True):
        """
        预处理数据：处理缺失值并准备训练数据，包括数据增强

        参数:
        add_poly_features: 是否添加多项式特征
        add_interactions: 是否添加交互特征
        augment_data: 是否进行数据增强

        返回:
        X: 处理后的特征数组
        y: 目标变量数组
        """
        print(f"\n=== 数据预处理 ===")

        # 删除包含缺失值的行，确保数据质量
        original_size = self.features_df.shape[0]
        self.features_df = self.features_df.dropna()
        new_size = self.features_df.shape[0]
        print(f"删除缺失值: 原始样本数 {original_size}, 处理后样本数 {new_size}, 删除了 {original_size - new_size} 行")

        # 将目标数据与特征数据对齐（确保它们有相同的索引）
        self.target = self.target_df.loc[self.features_df.index]
        print(f"对齐后目标数据形状: {self.target.shape}")

        # 特征扩展
        X_df = self.features_df.copy()

        # 添加交互特征
        if add_interactions:
            X_df = self.add_interaction_features(X_df)

        # 保存特征列名，用于后续分析和解释模型
        self.feature_names = list(X_df.columns)

        # 提取特征和标签为numpy数组
        X = X_df.values  # 特征矩阵
        y = self.target.values.reshape(-1, 1)  # 目标变量，确保是2D数组

        # 特征缩放
        X = self.scaler.fit_transform(X)

        # 添加多项式特征
        if add_poly_features:
            X = self.add_polynomial_features(X, degree=2)

        # 数据增强
        if augment_data:
            X, y = self.augment_with_noise(X, y, noise_level=0.03)

        print(f"最终特征矩阵形状: {X.shape}, 目标变量形状: {y.shape}")
        return X, y

    def split_test_set(self, X, y, test_size=0.2, random_state=42):
        """
        将数据集划分为训练+验证集和测试集

        参数:
        X: 特征矩阵
        y: 目标变量
        test_size: 测试集比例
        random_state: 随机种子

        返回:
        X_train_val, X_test, y_train_val, y_test: 划分后的数据集
        """
        print(f"\n=== 划分测试集 ===")

        # 划分测试集
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"总样本数: {len(X)}")
        print(f"训练+验证集: {X_train_val.shape}, {y_train_val.shape} ({len(X_train_val) / len(X):.2%})")
        print(f"测试集: {X_test.shape}, {y_test.shape} ({len(X_test) / len(X):.2%})")

        return X_train_val, X_test, y_train_val, y_test

    def split_train_val(self, X_train_val, y_train_val, val_size=0.2, random_state=42):
        """
        将训练+验证集进一步划分为训练集和验证集

        参数:
        X_train_val: 训练+验证特征
        y_train_val: 训练+验证目标
        val_size: 验证集占训练+验证集的比例
        random_state: 随机种子

        返回:
        X_train, X_val, y_train, y_val: 划分后的训练集和验证集
        """
        print(f"\n=== 划分训练集和验证集 ===")

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_state
        )

        print(f"训练+验证集总样本数: {len(X_train_val)}")
        print(f"训练集: {X_train.shape}, {y_train.shape} ({len(X_train) / len(X_train_val):.2%})")
        print(f"验证集: {X_val.shape}, {y_val.shape} ({len(X_val) / len(X_train_val):.2%})")

        return X_train, X_val, y_train, y_val

    def split_data(self, X, y, test_size=0.2, val_size=0.2, random_state=42):
        """
        划分数据集为训练集、验证集和测试集

        参数:
        X: 特征矩阵
        y: 目标变量
        test_size: 测试集比例（占总数据的比例）
        val_size: 验证集比例（占剩余数据的比例）
        random_state: 随机种子

        返回:
        X_train, X_val, X_test, y_train, y_val, y_test: 划分后的数据集
        """
        print(f"\n=== 划分数据集 ===")

        # 首先划分出测试集（占总数据的20%）
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 从剩余数据中划分出验证集（占剩余数据的20%）
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_state
        )

        # 打印数据集大小
        print("\n数据划分结果:")
        print(f"总样本数: {len(X)}")
        print(f"训练集形状: {X_train.shape}, {y_train.shape}")
        print(f"验证集形状: {X_val.shape}, {y_val.shape}")
        print(f"测试集形状: {X_test.shape}, {y_test.shape}")

        # 计算并打印各集合的比例
        total = len(X)
        print(f"\n数据集比例:")
        print(f"训练集: {len(X_train) / total:.2%}")
        print(f"验证集: {len(X_val) / total:.2%}")
        print(f"测试集: {len(X_test) / total:.2%}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def transform_test_data(self, X_test):
        """
        对测试数据应用与训练数据相同的转换

        参数:
        X_test: 测试特征数据

        返回:
        转换后的测试特征
        """
        print(f"测试集特征工程: 原始特征数 {X_test.shape[1]}")

        # 首先应用标准化（使用训练集拟合的标准化器）
        # 注意：这一步必须在添加交互特征之前进行
        X_test_scaled = self.scaler.transform(X_test)

        # 如果需要添加交互特征
        if hasattr(self, 'feature_names') and len(self.feature_names) > 0:
            # 创建DataFrame以计算交互特征
            X_test_df = pd.DataFrame(X_test_scaled, columns=[f'feature_{i}' for i in range(X_test_scaled.shape[1])])

            # 添加交互特征（与训练集相同的特征交互方式）
            col_names = X_test_df.columns
            for i in range(len(col_names)):
                for j in range(i + 1, len(col_names)):
                    # 创建乘积交互项
                    col_name = f"{col_names[i]}_x_{col_names[j]}"
                    X_test_df[col_name] = X_test_df[col_names[i]] * X_test_df[col_names[j]]

            X_test_processed = X_test_df.values
            print(f"添加交互特征后测试集特征数: {X_test_processed.shape[1]}")
        else:
            X_test_processed = X_test_scaled

        # 应用多项式特征（如果已初始化）
        if self.poly is not None:
            X_test_transformed = self.poly.transform(X_test_scaled)
            print(f"应用多项式特征后测试集特征数: {X_test_transformed.shape[1]}")
            return X_test_transformed

        return X_test_processed


# 自定义数据集类，继承自PyTorch的Dataset类
# 这个类将numpy数组转换为PyTorch可以处理的格式
class PrecipitationDataset(Dataset):
    def __init__(self, features, targets):
        """
        初始化数据集
        :param features: 特征数据
        :param targets: 目标数据
        """
        # 处理异常值
        self.features = self.handle_outliers(features)
        self.targets = self.handle_outliers(targets)

        # 转换为张量
        self.features = torch.FloatTensor(self.features)
        self.targets = torch.FloatTensor(self.targets)

    def handle_outliers(self, data, threshold=3):
        """使用IQR方法处理异常值"""
        if isinstance(data, torch.Tensor):
            data = data.numpy()

        # 计算四分位数
        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)
        IQR = Q3 - Q1

        # 定义异常值边界
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # 将异常值替换为边界值
        data = np.clip(data, lower_bound, upper_bound)
        return data

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
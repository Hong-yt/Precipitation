import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
import os

# 忽略警告信息，使输出更清晰
warnings.filterwarnings("ignore")

# 设置随机种子以确保结果可重复
# 这样每次运行代码时得到的结果都是一样的，便于调试和比较不同参数设置的效果
torch.manual_seed(42)
np.random.seed(42)

# 解决可能的DLL冲突问题（针对Windows环境中PyTorch和其他库的兼容性问题）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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

    def load_data(self):
        """
        加载特征数据和目标数据

        返回:
        features_df: 包含所有输入特征的DataFrame
        target_df: 包含目标变量(R值)的DataFrame
        """

        print(f"\n=== 正在加载数据 ===")

        # 加载特征数据，忽略前两列（行列号）
        self.features_df = pd.read_csv(self.features_path).iloc[:, 2:]

        # 加载目标数据（实际站点的 R 值），忽略前两列（如果目标数据也有行列号）
        self.target_df = pd.read_csv(self.target_path).iloc[:, 2:]

        # 打印数据形状，便于检查数据加载是否正确
        print("\n数据加载结果:")
        print(f"特征数据前5行:\n{self.features_df.head()}")
        print(f"目标数据前5行:\n{self.target_df.head()}")
        print(f"shape of feature: {self.features_df.shape}")  # 预期形状大约为 (8000, 20)
        print(f"shape of target: {self.target_df.shape}")  # 预期形状大约为 (8000, 1)

        return self.features_df, self.target_df

    def preprocess_data(self):
        """
        预处理数据：处理缺失值并准备训练数据

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

        # 提取特征和标签为numpy数组
        X = self.features_df.values  # 特征矩阵
        y = self.target.values.reshape(-1, 1)  # 目标变量，确保是2D数组
        print(f"最终特征矩阵形状: {X.shape}, 目标变量形状: {y.shape}")

        # 保存特征列名，用于后续分析和解释模型
        self.feature_names = list(self.features_df.columns)

        return X, y

    def split_data(self, X, y, test_size=0.2, val_size=0.2):
        """
        将数据分为训练集、验证集和测试集

        参数:
        X: 特征数据
        y: 目标数据
        test_size: 测试集比例，默认为总数据的20%
        val_size: 验证集比例，默认为剩余数据(训练+验证)的20%

        返回:
        (X_train, X_val, X_test, y_train, y_val, y_test): 划分后的数据集
        """

        print(f"\n=== 划分数据集 ===")
        total_samples = X.shape[0]

        # 首先划分出测试集（这部分数据在训练过程中完全不使用，仅用于最终评估）
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # 从剩余数据中划分出验证集
        # val_size相对于train_val数据的比例需要重新计算
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, random_state=42
        )

        # 打印数据集大小，便于检查划分是否合理
        print("\n数据划分结果:")
        print(f"总样本数: {total_samples}")
        print(f"训练集形状: {X_train.shape}, {y_train.shape}")  # 约64%的数据
        print(f"验证集形状: {X_val.shape}, {y_val.shape}")  # 约16%的数据
        print(f"测试集形状: {X_test.shape}, {y_test.shape}")  # 约20%的数据

        return X_train, X_val, X_test, y_train, y_val, y_test


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


# 定义深度神经网络模型
# 这是整个代码的核心部分，定义了网络的结构和前向传播逻辑
class PrecipitationDNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.2):
        """
        初始化深度神经网络模型

        参数:
        input_size: 输入特征的数量（对应您数据中的特征数，20个）
        hidden_sizes: 各隐藏层的神经元数量，默认为[128, 64, 32]
        dropout_rate: Dropout比率，用于防止过拟合，默认为0.2（20%的神经元会被随机关闭）
        """
        super(PrecipitationDNN, self).__init__()
        print(f"初始化神经网络，输入特征数: {input_size}")

        # 构建网络层
        layers = []
        prev_size = input_size

        # 添加隐藏层
        # 每个隐藏层包括：线性层 -> ReLU激活 -> 批归一化 -> Dropout
        for hidden_size in hidden_sizes:
            # 线性层（全连接层）
            layers.append(nn.Linear(prev_size, hidden_size))
            # ReLU激活函数，引入非线性变换
            layers.append(nn.ReLU())
            # 批归一化层，加速训练并提高稳定性
            layers.append(nn.BatchNorm1d(hidden_size))
            # Dropout层，防止过拟合
            layers.append(nn.Dropout(dropout_rate))
            # 更新前一层大小，用于下一层的输入大小
            prev_size = hidden_size

        # 输出层，输出单个值（预测的R值）
        layers.append(nn.Linear(prev_size, 1))

        # 使用Sequential容器将所有层组合在一起
        self.network = nn.Sequential(*layers)

        print("神经网络构建完成！")

    def forward(self, x):
        """
        前向传播函数，定义数据通过网络的流动方式

        参数:
        x: 输入数据

        返回:
        网络的输出（预测的R值）
        """
        return self.network(x)


# 训练和评估模型的类
# 这个类负责模型的训练、评估和结果可视化
class PrecipitationModelTrainer:
    def __init__(self, model, dataset_name, device):
        """
        初始化模型训练器

        参数:
        model: 神经网络模型
        dataset_name: 数据集名称（用于保存结果）
        device: 训练设备（CPU或GPU）
        """
        print("初始化模型训练器...")

        self.model = model
        self.dataset_name = dataset_name
        self.device = device
        # 将模型移至指定设备（CPU或GPU）
        self.model.to(device)

        # 设置损失函数和优化器
        # MSE（均方误差）损失函数，适用于回归问题
        self.criterion = nn.MSELoss()
        # Adam优化器，自适应学习率，一般表现良好
        # weight_decay参数提供L2正则化，防止过拟合
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        # 学习率调度器，当验证损失不再下降时，降低学习率
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

    def train(self, train_loader, val_loader, epochs=200):
        """
        训练模型

        参数:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数，默认200轮

        返回:
        train_losses: 训练损失历史
        val_losses: 验证损失历史
        """

        print("开始训练...")

        train_losses = []  # 记录每个epoch的训练损失
        val_losses = []  # 记录每个epoch的验证损失
        best_val_loss = float('inf')  # 记录最佳验证损失
        best_model_state = None  # 保存表现最佳的模型参数

        patience = 20
        no_improve_epochs = 0

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()  # 将模型设置为训练模式（启用dropout等）
            train_loss = 0

            # 遍历训练数据批次
            for features, targets in train_loader:
                # 将数据移到指定设备（CPU或GPU）
                features, targets = features.to(self.device), targets.to(self.device)

                # 前向传播
                outputs = self.model(features)
                # 计算损失
                loss = self.criterion(outputs, targets)

                # 反向传播和优化
                self.optimizer.zero_grad()  # 清除之前的梯度
                loss.backward()  # 计算梯度
                self.optimizer.step()  # 更新参数

                train_loss += loss.item()  # 累加批次损失

            # 计算平均训练损失（除以批次数）
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # 验证阶段
            self.model.eval()  # 将模型设置为评估模式（禁用dropout等）
            val_loss = 0

            # 验证时不需要计算梯度，节省内存
            with torch.no_grad():
                for features, targets in val_loader:
                    features, targets = features.to(self.device), targets.to(self.device)
                    # 计算模型输出
                    outputs = self.model(features)
                    # 计算验证损失
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()

            # 计算平均验证损失
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']

            # 学习率调度器根据验证损失调整学习率
            self.scheduler.step(val_loss)

            # 如果当前模型在验证集上表现更好，则保存模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= patience:
                print(f'早停: {epoch + 1} 轮后验证损失未改善')
                break

            # 每10个epoch打印一次训练进度
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, 学习率: {current_lr:.6f}')

        # 训练结束后，加载表现最佳的模型参数
        self.model.load_state_dict(best_model_state)

        print(f"训练结束！最佳验证损失: {best_val_loss:.4f}，已加载最佳模型参数。")

        return train_losses, val_losses

    def evaluate(self, test_loader):
        """
        评估模型在测试集上的表现

        参数:
        test_loader: 测试数据加载器

        返回:
        predictions: 模型预测值
        actuals: 真实值
        """
        # 设置为评估模式
        self.model.eval()
        predictions = []
        actuals = []

        # 不计算梯度
        with torch.no_grad():
            for features, targets in test_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                # 计算模型输出
                outputs = self.model(features)

                # 收集预测值和真实值
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.cpu().numpy())

        # 转换为numpy数组并展平
        predictions = np.array(predictions).flatten()
        actuals = np.array(actuals).flatten()

        # 计算评估指标
        correlation = np.corrcoef(predictions, actuals)[0, 1]  # 相关系数 (R)
        r2 = r2_score(actuals, predictions)  # 决定系数 (R²)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))  # 均方根误差 (RMSE)
        mae = mean_absolute_error(actuals, predictions)  # 平均绝对误差 (MAE)
        mse = mean_squared_error(actuals, predictions)  # 均方误差 (MSE)

        # 打印评估结果
        print(f"\n测试集评估结果:")
        print(f"相关系数 (R): {correlation:.4f}")
        print(f"决定系数 (R²): {r2:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f}")
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"平均绝对误差 (MAE): {mae:.4f}")

        return predictions, actuals

    # def cross_validation(self, X, y, n_splits=5, batch_size=32, epochs=100):
    #     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    #     cv_scores = []
    #
    #     for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    #         print(f"\n==== 第 {fold + 1} 折 ====")
    #
    #         # 标准化处理（每个折独立进行）
    #         X_train, X_val = X[train_idx], X[val_idx]
    #         y_train, y_val = y[train_idx], y[val_idx]
    #
    #         scaler = StandardScaler()
    #         X_train = scaler.fit_transform(X_train)
    #         X_val = scaler.transform(X_val)
    #
    #         # 创建数据加载器
    #         train_set = PrecipitationDataset(X_train, y_train)
    #         val_set = PrecipitationDataset(X_val, y_val)
    #         train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    #         val_loader = DataLoader(val_set, batch_size=batch_size)
    #
    #         # 初始化模型和优化器
    #         model = PrecipitationDNN(X.shape[1]).to(self.device)
    #         optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    #
    #         # 训练循环
    #         model.train()
    #         for epoch in range(epochs):
    #             epoch_loss = 0
    #             for features, targets in train_loader:
    #                 features, targets = features.to(self.device), targets.to(self.device)
    #                 optimizer.zero_grad()
    #                 outputs = model(features)
    #                 loss = self.criterion(outputs, targets)
    #                 loss.backward()
    #                 optimizer.step()
    #                 epoch_loss += loss.item()
    #             epoch_loss /= len(train_loader)
    #             if (epoch + 1) % 20 == 0:
    #                 print(f"Epoch {epoch + 1}/{epochs} 损失: {epoch_loss:.4f}")
    #
    #         # 验证评估
    #         model.eval()
    #         preds, truths = [], []
    #         with torch.no_grad():
    #             for features, targets in val_loader:
    #                 features = features.to(self.device)
    #                 outputs = model(features).cpu().numpy()
    #                 preds.extend(outputs)
    #                 truths.extend(targets.numpy())
    #         correlation = np.corrcoef(preds, truths)[0, 1]
    #         cv_scores.append(correlation)
    #         print(f"验证R值: {correlation:.4f}")
    #
    #     print(f"\n平均交叉验证R值: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    #     return cv_scores

    def visualize_results(self, train_losses, val_losses, predictions, actuals, output_dir):
        """
        可视化训练结果

        参数:
        train_losses: 训练损失历史
        val_losses: 验证损失历史
        predictions: 预测值
        actuals: 真实值
        """


        # 创建DataFrame来保存预测值和真实值
        results_df = pd.DataFrame({
            'Predictions': predictions,
            'Actuals': actuals
        })
        # 将结果保存为CSV文件
        output_file = os.path.join(output_dir, f"{self.dataset_name}_predictions_vs_actuals.csv")
        results_df.to_csv(output_file, index=False)
        print(f"预测结果已保存到: {output_file}")

        # 创建一个大的图形
        plt.figure(figsize=(15, 10))

        # 绘制损失曲线
        plt.subplot(2, 1, 1)  # 2行1列的第1个子图
        plt.plot(train_losses, label='训练损失')
        plt.plot(val_losses, label='验证损失')
        plt.xlabel('训练轮数')
        plt.ylabel('损失值')
        plt.title(f'{self.dataset_name} - 训练与验证损失曲线')
        plt.legend()

        # 绘制预测vs真实值散点图
        plt.subplot(2, 1, 2)  # 2行1列的第2个子图
        plt.scatter(actuals, predictions, alpha=0.5)

        # 添加理想线(y=x)，完美预测的情况下，所有点应该位于这条线上
        min_val = min(min(actuals), min(predictions))
        max_val = max(max(actuals), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.xlabel('实际R值')
        plt.ylabel('预测R值')
        plt.title(f'{self.dataset_name} - 预测值与实际值对比')

        # 调整子图之间的间距
        plt.tight_layout()
        # 保存图像
        image_file = os.path.join(output_dir, f"{self.dataset_name}_dnn_results.png")

        plt.savefig(image_file)
        print(f"图像已保存到: {image_file}")

        plt.show()


# 主函数：运行完整的训练和评估流程
def main():
    """主函数，执行完整的数据处理、模型训练和评估流程"""
    # 设置数据集名称和路径
    # 注意：这里需要替换为实际的文件路径
    dataset_name = "imerg"  # 示例数据集名称
    features_path = f"D:/data/{dataset_name}_features.csv"
    target_path = f"D:/data/{dataset_name}_target.csv"

    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    # 检测是否可以使用GPU，若可用则使用GPU加速训练
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"当前GPU内存使用情况: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

    # ==== 数据加载和预处理 ====
    # 初始化数据处理器
    print("\n=== 数据加载与预处理阶段开始 ===")

    data_processor = PrecipitationDataProcessor(dataset_name, features_path, target_path)

    # 加载原始数据
    features_df, target_df = data_processor.load_data()

    # 预处理数据（处理缺失值和对齐数据）
    X, y = data_processor.preprocess_data()

    # 划分数据集（此时不进行标准化）
    X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data(X, y)

    # ==== 标准化处理 ====
    # 注意：标准化应该在数据划分后进行，使用训练集的统计量来标准化所有数据集
    print("\n=== 数据标准化 ===")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # 只在训练集上拟合标准化器
    X_val = scaler.transform(X_val)  # 使用训练集的均值和标准差标准化验证集
    X_test = scaler.transform(X_test)  # 使用训练集的均值和标准差标准化测试集
    print(f"标准化参数 - 均值: {scaler.mean_}, 标准差: {scaler.scale_}")

    # ==== 数据准备 ====
    # 创建PyTorch数据集
    print("\n=== 准备PyTorch数据加载器 ===")
    train_dataset = PrecipitationDataset(X_train, y_train)
    val_dataset = PrecipitationDataset(X_val, y_val)
    test_dataset = PrecipitationDataset(X_test, y_test)

    print(f"训练数据集样本数: {len(train_dataset)}")
    print(f"验证数据集样本数: {len(val_dataset)}")
    print(f"测试数据集样本数: {len(test_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    print(f"\n数据加载器信息:")
    print(f"训练批次数: {len(train_loader)} (批量大小: 32)")
    print(f"验证批次数: {len(val_loader)}")
    print(f"测试批次数: {len(test_loader)}")

    # ==== 模型初始化 ====
    print("\n=== 初始化模型 ===")
    input_size = X_train.shape[1]  # 输入特征维度（20）
    print(f"输入特征数量: {input_size}")

    model = PrecipitationDNN(
        input_size=input_size,
        hidden_sizes=[128, 64, 32],  # 根据特征数量调整隐藏层大小
        dropout_rate=0.2
    )
    print("\n模型结构:")
    print(model)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n可训练参数总数: {total_params:,}")

    # 初始化训练器
    trainer = PrecipitationModelTrainer(model, dataset_name, device)

    # ==== 训练配置 ====
    print("\n=== 训练配置 ===")
    print(f"优化器: {type(trainer.optimizer).__name__}")
    print(f"初始学习率: {trainer.optimizer.param_groups[0]['lr']}")
    print(f"损失函数: {type(trainer.criterion).__name__}")

    # ==== 模型训练 ====
    print("\n=== 开始训练 ===")
    print(f"最大训练轮数: 200")
    print(f"早停机制: 连续20轮验证损失未改善则停止")
    train_losses, val_losses = trainer.train(train_loader, val_loader, epochs=200)

    # ==== 模型评估 ====
    print("\n=== 测试集评估 ===")
    predictions, actuals = trainer.evaluate(test_loader)
    print(f"预测结果形状: {predictions.shape}, 实际值形状: {actuals.shape}")

    # ==== 结果可视化 ====
    print("\n=== 保存结果 ===")
    print(f"输出目录: {output_dir}")

    trainer.visualize_results(train_losses, val_losses, predictions, actuals, output_dir)

    print("\n=== 所有流程执行完毕 ===")

if __name__ == "__main__":
    main()
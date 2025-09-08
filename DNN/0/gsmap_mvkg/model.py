import torch
import torch.nn as nn
import copy


class PrecipitationDNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32], dropout_rate=0.2, activation='relu'):
        """
        初始化深度神经网络模型

        参数:
        input_size: 输入特征的数量
        hidden_sizes: 各隐藏层的神经元数量，默认为[256, 128, 64]
        dropout_rate: Dropout比率，用于防止过拟合，默认为0.2
        activation: 激活函数类型，支持'relu', 'leaky_relu', 'elu'
        """
        super(PrecipitationDNN, self).__init__()
        print(f"初始化神经网络，输入特征数: {input_size}")

        # 选择激活函数
        self.activation_name = activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            print(f"不支持的激活函数 {activation}，使用ReLU替代")
            self.activation = nn.ReLU()
            self.activation_name = 'relu'

        print(f"使用激活函数: {self.activation_name}")

        # 特征提取部分
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            self.activation,
            nn.Dropout(dropout_rate)
        )

        # 主干网络 - 带残差连接
        self.layers = nn.ModuleList()
        self.shortcuts = nn.ModuleList()

        for i in range(len(hidden_sizes) - 1):
            # 每个残差块有两个线性层
            layer = nn.Sequential(
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                nn.BatchNorm1d(hidden_sizes[i + 1]),
                self.activation,
                nn.Dropout(dropout_rate * (1 - 0.1 * i)),  # 逐层减小dropout率
                nn.Linear(hidden_sizes[i + 1], hidden_sizes[i + 1]),
                nn.BatchNorm1d(hidden_sizes[i + 1]),
                self.activation,
                nn.Dropout(dropout_rate * (1 - 0.1 * i))
            )
            self.layers.append(layer)

            # 如果维度不匹配，添加一个投影快捷连接
            if hidden_sizes[i] != hidden_sizes[i + 1]:
                shortcut = nn.Sequential(
                    nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                    nn.BatchNorm1d(hidden_sizes[i + 1])
                )
            else:
                shortcut = nn.Identity()
            self.shortcuts.append(shortcut)

        # Attention层 - 为最后的特征添加注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 4),
            self.activation,
            nn.Linear(hidden_sizes[-1] // 4, hidden_sizes[-1]),
            nn.Sigmoid()
        )

        # 输出层
        self.final_layers = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 32),
            self.activation,
            nn.Linear(32, 1)
        )

        # 初始化权重
        self.apply(self._init_weights)

        # 打印模型参数量
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"可训练参数总数: {total_params:,}")
        print(f"网络深度: {len(hidden_sizes) + 2} 层")  # 包括输入层和输出层
        print("神经网络构建完成！")

    def _init_weights(self, m):
        """初始化网络权重"""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                    nonlinearity='relu' if self.activation_name == 'relu' else 'leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播函数，定义数据通过网络的流动方式

        参数:
        x: 输入数据

        返回:
        网络的输出（预测的R值）
        """
        # 特征提取
        x = self.feature_extractor(x)

        # 通过残差块
        for layer, shortcut in zip(self.layers, self.shortcuts):
            residual = shortcut(x)
            x = layer(x)
            x = x + residual  # 残差连接

        # 应用注意力机制
        attn_weights = self.attention(x)
        x = x * attn_weights  # 特征加权

        # 最终输出层
        x = self.final_layers(x)

        return x

    @staticmethod
    def create_model_for_fold(input_size, hidden_sizes, dropout_rate, fold_idx, activation='relu'):
        """
        为特定的交叉验证折创建一个新模型

        参数:
        input_size: 输入特征维度
        hidden_sizes: 隐藏层配置
        dropout_rate: Dropout率
        fold_idx: 当前折索引
        activation: 激活函数类型

        返回:
        新的模型实例
        """
        print(f"为第{fold_idx}折创建模型")
        return PrecipitationDNN(input_size, hidden_sizes, dropout_rate, activation)


def init_model(input_size, hidden_sizes, dropout_rate, device, activation='relu'):
    """
    初始化模型并将其移至指定设备

    参数:
    input_size: 输入特征维度
    hidden_sizes: 隐藏层神经元数量列表
    dropout_rate: Dropout比率
    device: 训练设备
    activation: 激活函数类型

    返回:
    model: 初始化并移至设备的模型
    """
    print("\n=== 初始化模型 ===")

    print(f"输入特征数量: {input_size}")
    print(f"隐藏层配置: {hidden_sizes}")
    print(f"Dropout率: {dropout_rate}")
    print(f"激活函数: {activation}")

    model = PrecipitationDNN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate,
        activation=activation
    )

    model.to(device)
    print(f"模型已移至设备: {device}")
    print("\n模型结构:")
    print(model)

    return model
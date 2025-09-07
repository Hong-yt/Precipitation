import torch
import torch.nn as nn


class PrecipitationDNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 96, 64, 32], dropout_rate=0.2):
        """
        初始化深度神经网络模型

        参数:
        input_size: 输入特征的数量
        hidden_sizes: 各隐藏层的神经元数量，默认为[256, 128, 96, 64, 32]
        dropout_rate: Dropout比率，用于防止过拟合，默认为0.2
        """
        super(PrecipitationDNN, self).__init__()
        print(f"初始化神经网络，输入特征数: {input_size}")

        # 构建网络层
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.input_bn = nn.BatchNorm1d(hidden_sizes[0])
        self.input_relu = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout_rate)

        # 隐藏层网络
        self.layers = nn.ModuleList()
        # 残差连接列表
        self.residual_layers = nn.ModuleList()

        for i in range(1, len(hidden_sizes)):
            # 主网络路径
            layer_block = nn.Sequential(
                nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_sizes[i]),
                nn.Dropout(dropout_rate)
            )
            self.layers.append(layer_block)

            # 残差连接（如果维度匹配）
            if hidden_sizes[i - 1] != hidden_sizes[i]:
                # 当维度不匹配时，添加转换层
                res_layer = nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])
            else:
                # 维度匹配时使用恒等映射
                res_layer = nn.Identity()
            self.residual_layers.append(res_layer)

        # 输出层
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)

        # 打印模型参数量
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"可训练参数总数: {total_params:,}")
        print("神经网络构建完成！")

    def forward(self, x):
        """
        前向传播函数，定义数据通过网络的流动方式

        参数:
        x: 输入数据

        返回:
        网络的输出（预测的R值）
        """
        # 输入层
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = self.input_relu(x)
        x = self.input_dropout(x)

        # 隐藏层（带残差连接）
        for i, (layer, res_layer) in enumerate(zip(self.layers, self.residual_layers)):
            identity = res_layer(x)
            x = layer(x)
            x = x + identity  # 残差连接

        # 输出层
        x = self.output_layer(x)
        return x


def init_model(input_size, hidden_sizes, dropout_rate, device):
    """
    初始化模型并将其移至指定设备

    参数:
    input_size: 输入特征维度
    hidden_sizes: 隐藏层神经元数量列表
    dropout_rate: Dropout比率
    device: 训练设备

    返回:
    model: 初始化并移至设备的模型
    """
    print("\n=== 初始化模型 ===")

    print(f"输入特征数量: {input_size}")
    print(f"隐藏层配置: {hidden_sizes}")
    print(f"Dropout率: {dropout_rate}")

    model = PrecipitationDNN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate
    )

    model.to(device)
    print(f"模型已移至设备: {device}")
    print("\n模型结构:")
    print(model)

    return model
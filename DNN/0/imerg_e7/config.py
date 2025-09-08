import os
import torch
import warnings
import numpy as np
from pathlib import Path

# 忽略警告信息，使输出更清晰
warnings.filterwarnings("ignore")

# 解决可能的DLL冲突问题（针对Windows环境中PyTorch和其他库的兼容性问题）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Config:
    # 数据集配置
    DATASET_NAME = "imerg_e7"
    FEATURES_PATH = f"D:/hongyouting/data/dnn/merge/features/{DATASET_NAME}_features.xlsx"
    TARGET_PATH = f"D:/hongyouting/data/dnn/merge/target/{DATASET_NAME}_target.xlsx"

    # 输出目录配置
    OUTPUT_DIR = 'D:/hongyouting/result/dnn/0/imerg_e7'
    TRAIN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'train')
    TEST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'test')
    CV_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'cv')  # 添加交叉验证结果目录
    HYPERPARAM_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'hyperparams')  # 超参数优化结果目录

    # 交叉验证配置
    CV_FOLDS = 5  # 5折交叉验证
    CV_SHUFFLE = True  # 交叉验证前随机打乱数据

    # 数据处理配置
    OUTLIER_REMOVAL = True  # 是否移除离群值
    OUTLIER_THRESHOLD = 3.0  # z-score阈值，超过此值的样本被视为离群值
    LOG_TRANSFORM = True  # 是否对目标变量进行对数变换
    LOG_OFFSET = 0.01  # 对数变换的偏移量，避免log(0)
    FEATURE_SELECTION = False  # 是否进行特征选择，设置为False保留所有原始特征
    FEATURE_IMPORTANCE_THRESHOLD = 0.01  # 特征重要性阈值

    # 特征工程配置
    FEATURE_ENGINEERING = True  # 是否应用特征工程
    NONLINEAR_TRANSFORM = True  # 是否应用非线性变换（sqrt、square、log等）
    INTERACTION_FEATURES = True  # 是否创建交互特征
    INTERACTION_DEGREE = 2  # 交互特征的多项式次数
    STATISTICAL_FEATURES = True  # 是否创建统计特征（均值、标准差等）

    # 时序特征配置（如果数据含有时序成分）
    TIME_LAG_FEATURES = False  # 是否生成时间滞后特征
    LAG_COLUMNS = [0, 1, 2, 3, 4]  # 要为哪些列创建滞后特征（默认前5列）
    LAG_STEPS = [1, 2, 3]  # 滞后步长

    # 数据增强配置
    DATA_AUGMENTATION = True  # 是否进行数据增强
    NOISE_FACTOR = 0.01  # 加到样本上的随机噪声的幅度因子
    AUG_SAMPLES = None  # 增强生成的样本数，None表示与原始样本数相同

    # 超参数优化配置
    HYPERPARAMETER_TUNING = True  # 是否进行超参数优化
    HYPERPARAMETER_SEARCH_METHOD = 'grid'  # 'grid', 'random', 'bayesian'

    # 超参数网格搜索范围
    GRID_PARAMS = {
        'hidden_sizes': [
            # [256, 128, 64],
            [256, 128, 64, 32],
            # [128, 128, 128],
            # [256, 256, 128, 64, 32],
            # [512, 512, 256, 128, 64]  # 添加更深的网络结构
        ],
        'dropout_rate': [0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'batch_size': [32, 64],
        'weight_decay': [1e-4, 1e-5, ],
        'optimizer': ['adam'],
        'activation': ['relu'],  # 添加mish激活函数
        'patience': [15]
    }

    # 特征工程相关的超参数
    FEATURE_ENGINEERING_PARAMS = {
        'interaction_degree': [2],
        'nonlinear_transform': [True],
        'statistical_features': [True],
        'noise_factor': [0.01]
    }

    # 网格搜索优化时的最大组合数，超过此数量将进行随机采样
    MAX_GRID_COMBINATIONS = 30  # 增加到25以探索更多组合

    # 数据划分配置（不再使用，但保留以便向后兼容）
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2

    # 模型配置
    HIDDEN_SIZES = [512, 256, 128, 64]  # 调整为更复杂的网络
    DROPOUT_RATE = 0.2
    ACTIVATION = 'relu'  # 激活函数类型

    # 训练配置
    BATCH_SIZE = 32
    EPOCHS = 400  # 增加训练轮数
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    PATIENCE = 30  # 增加早停耐心值
    OPTIMIZER = 'adam'  # 优化器类型

    # 设备配置
    DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # 随机种子配置
    SEED = 42
    TORCH_SEED = 42
    NUMPY_SEED = 42

    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.TRAIN_OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.TEST_OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CV_OUTPUT_DIR, exist_ok=True)  # 创建交叉验证结果目录
        os.makedirs(cls.HYPERPARAM_OUTPUT_DIR, exist_ok=True)  # 创建超参数优化结果目录

    @classmethod
    def update_params(cls, params_dict):
        """更新配置参数"""
        for key, value in params_dict.items():
            if hasattr(cls, key.upper()):
                setattr(cls, key.upper(), value)
                print(f"更新参数: {key.upper()} = {value}")
            else:
                print(f"警告: 参数 {key} 不存在于配置中")
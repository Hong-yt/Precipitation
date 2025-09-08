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
    DATASET_NAME = "gsmap_mvkg"
    FEATURES_PATH = f"D:/hongyouting/data/dnn/merge/features/{DATASET_NAME}_features.xlsx"
    TARGET_PATH = f"D:/hongyouting/data/dnn/merge/target/{DATASET_NAME}_target.xlsx"

    # 输出目录配置
    OUTPUT_DIR = "D:/hongyouting/result/dnn/gsmap_mvkg"
    TRAIN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "train")
    TEST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "test")
    CV_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "cv")

    # 数据划分配置
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2

    # 交叉验证配置
    N_SPLITS = 5  # 交叉验证折数
    BATCH_SIZES = [32, 64]  # 要搜索的batch size列表
    DROPOUT_RATES = [0.2, 0.3, 0.4]  # 要搜索的dropout rate列表
    LEARNING_RATES = [0.0001, 0.0005, 0.001]  # 要搜索的学习率列表
    CV_EPOCHS = 100  # 交叉验证时的训练轮数

    # 模型配置
    HIDDEN_SIZES = [256, 128, 64]
    EPOCHS = 300  # 最终训练轮数
    BATCH_SIZE = 64  # 默认batch size
    LEARNING_RATE = 0.0005  # 默认学习率
    DROPOUT_RATE = 0.3  # 默认dropout rate
    WEIGHT_DECAY = 0.001  # L2正则化系数

    # 训练配置
    PATIENCE = 15  # 早停耐心值
    SEED = 42  # 随机种子

    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 随机种子配置
    TORCH_SEED = 42
    NUMPY_SEED = 42

    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.TRAIN_OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.TEST_OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CV_OUTPUT_DIR, exist_ok=True)
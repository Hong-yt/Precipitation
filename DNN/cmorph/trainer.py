import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from config import Config
import matplotlib.pyplot as plt


class PrecipitationModelTrainer:
    def __init__(self, model, dataset_name, device, optimizer_name='adam', lr=0.001, weight_decay=1e-5):
        """
        初始化模型训练器

        参数:
        model: 待训练的模型
        dataset_name: 数据集名称
        device: 训练设备
        optimizer_name: 优化器名称，支持'adam', 'adamw', 'sgd'
        lr: 学习率
        weight_decay: 权重衰减系数
        """
        print("初始化模型训练器...")
        self.model = model
        self.dataset_name = dataset_name
        self.device = device
        self.model.to(device)

        # 确保使用MSE损失
        self.criterion = nn.MSELoss()

        # 选择优化器
        if optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
            print(f"使用Adam优化器，学习率={lr}, 权重衰减={weight_decay}")
        elif optimizer_name.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
            print(f"使用AdamW优化器，学习率={lr}, 权重衰减={weight_decay}")
        elif optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
            print(f"使用SGD优化器，学习率={lr}, 动量=0.9, 权重衰减={weight_decay}")
        else:
            print(f"不支持的优化器 {optimizer_name}，使用Adam替代")
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        # 训练状态跟踪
        self.best_val_loss = float('inf')
        self.early_stop_counter = 0
        self.current_epoch = 0

        # 保存模型检查点的路径
        self.checkpoints_dir = os.path.join(Config.TRAIN_OUTPUT_DIR, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def train(self, train_loader, val_loader, epochs=200, patience=20, log_interval=10, collect_predictions=False):
        """
        训练模型

        参数:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 最大训练轮数
        patience: 早停耐心值，连续多少轮验证损失不下降则停止训练
        log_interval: 每多少轮输出一次详细日志
        collect_predictions: 是否收集每个epoch的预测值用于可视化

        返回:
        train_losses: 训练损失历史
        val_losses: 验证损失历史
        val_pred_history: 如果collect_predictions=True，返回每个epoch的验证集预测值历史
        """
        print("开始训练...")
        train_losses = []
        val_losses = []
        train_metrics_history = []
        val_metrics_history = []
        val_pred_history = [] if collect_predictions else None
        best_model_state = None
        best_epoch = 0
        best_val_r2 = -float('inf')

        # 跟踪GPU内存使用情况（如果可用）
        if self.device.type == 'cuda':
            initial_memory = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"初始GPU内存使用: {initial_memory:.2f} MB")

        for epoch in range(epochs):
            self.current_epoch = epoch
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_predictions = []
            train_targets = []
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')

            for features, targets in train_pbar:
                features, targets = features.to(self.device), targets.to(self.device)

                # 前向传播
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                # 收集统计信息
                train_loss += loss.item()
                train_predictions.extend(outputs.detach().cpu().numpy())
                train_targets.extend(targets.cpu().numpy())
                train_pbar.set_postfix({'MSE loss': f'{loss.item():.4f}'})

            # 计算平均训练损失
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # 计算训练集指标
            train_predictions = np.array(train_predictions).flatten()
            train_targets = np.array(train_targets).flatten()
            train_metrics = {
                'R²': r2_score(train_targets, train_predictions),
                'RMSE': np.sqrt(mean_squared_error(train_targets, train_predictions)),
                'MAE': mean_absolute_error(train_targets, train_predictions)
            }
            train_metrics_history.append(train_metrics)

            # 验证阶段
            self.model.eval()
            val_loss = 0
            val_predictions = []
            val_targets = []
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]')

            with torch.no_grad():
                for features, targets in val_pbar:
                    features, targets = features.to(self.device), targets.to(self.device)
                    outputs = self.model(features)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()
                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
                    val_pbar.set_postfix({'MSE loss': f'{loss.item():.4f}'})

            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            # 计算验证集指标
            val_predictions = np.array(val_predictions).flatten()
            val_targets = np.array(val_targets).flatten()
            val_metrics = {
                'R²': r2_score(val_targets, val_predictions),
                'RMSE': np.sqrt(mean_squared_error(val_targets, val_predictions)),
                'MAE': mean_absolute_error(val_targets, val_predictions)
            }
            val_metrics_history.append(val_metrics)

            # 更新学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)

            # 保存最佳模型（基于验证集R²）
            if val_metrics['R²'] > best_val_r2:
                best_val_r2 = val_metrics['R²']
                best_model_state = self.model.state_dict().copy()
                best_epoch = epoch
                self.early_stop_counter = 0
                # 保存最佳模型
                self.save_model(best_model_state, epoch, val_metrics)
                print(f"✓ 发现新的最佳模型! 验证集 R² = {best_val_r2:.4f}")
            else:
                self.early_stop_counter += 1

            # 检查是否需要提前停止训练
            if self.early_stop_counter >= patience:
                print(f'早停: {epoch + 1} 轮后验证集 R² 未改善')
                break

            # 定期输出详细训练信息
            if (epoch + 1) % log_interval == 0 or epoch == 0 or epoch == epochs - 1:
                print(f'\nEpoch [{epoch + 1}/{epochs}]')
                print(f'学习率: {current_lr:.6f}')
                print('训练集:')
                print(f'  MSE损失: {train_loss:.4f}')
                for metric, value in train_metrics.items():
                    print(f'  {metric}: {value:.4f}')
                print('验证集:')
                print(f'  MSE损失: {val_loss:.4f}')
                for metric, value in val_metrics.items():
                    print(f'  {metric}: {value:.4f}')

                # 在使用GPU时报告内存使用情况
                if self.device.type == 'cuda':
                    current_memory = torch.cuda.memory_allocated() / (1024 ** 2)
                    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    print(f"GPU内存: 当前 {current_memory:.2f} MB, 峰值 {max_memory:.2f} MB")

            # 创建和保存训练进度图
            if epoch > 0 and (epoch + 1) % 20 == 0:
                self.plot_training_progress(train_losses, val_losses, train_metrics_history, val_metrics_history)

            # 如果需要收集预测值，保存本轮的预测
            if collect_predictions:
                val_pred_history.append(val_predictions)

        # 训练结束，加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n训练结束！最佳模型来自第 {best_epoch + 1} 轮, 验证集 R² = {best_val_r2:.4f}")
        else:
            print(f"\n训练结束！未找到优于初始模型的结果")

        # 保存训练历史
        self.save_training_history(train_losses, val_losses, train_metrics_history, val_metrics_history)

        # 创建并保存最终训练进度图
        self.plot_training_progress(train_losses, val_losses, train_metrics_history, val_metrics_history, final=True)

        return train_losses, val_losses, val_pred_history

    def save_model(self, model_state, epoch, metrics=None):
        """
        保存模型

        参数:
        model_state: 模型状态字典
        epoch: 当前轮数
        metrics: 评估指标
        """
        # 判断是否是交叉验证的模型
        if 'fold' in self.dataset_name:
            save_path = os.path.join(Config.CV_OUTPUT_DIR, f"{self.dataset_name}_model.pth")
        else:
            save_path = os.path.join(Config.TRAIN_OUTPUT_DIR, f"{self.dataset_name}_model.pth")

        save_dict = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }

        # 如果有评估指标，也保存
        if metrics is not None:
            save_dict['metrics'] = metrics

        torch.save(save_dict, save_path)
        print(f"模型已保存到: {save_path}")

    def plot_training_progress(self, train_losses, val_losses, train_metrics_history, val_metrics_history, final=False):
        """
        绘制并保存训练进度图

        参数:
        train_losses: 训练损失历史
        val_losses: 验证损失历史
        train_metrics_history: 训练集评估指标历史
        val_metrics_history: 验证集评估指标历史
        final: 是否是最终图表
        """
        epochs = range(1, len(train_losses) + 1)

        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 绘制损失图
        axes[0, 0].plot(epochs, train_losses, 'b-', label='训练损失')
        axes[0, 0].plot(epochs, val_losses, 'r-', label='验证损失')
        axes[0, 0].set_title('MSE损失')
        axes[0, 0].set_xlabel('轮次')
        axes[0, 0].set_ylabel('损失值')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # 绘制R²图
        train_r2 = [metrics['R²'] for metrics in train_metrics_history]
        val_r2 = [metrics['R²'] for metrics in val_metrics_history]
        axes[0, 1].plot(epochs, train_r2, 'b-', label='训练 R²')
        axes[0, 1].plot(epochs, val_r2, 'r-', label='验证 R²')
        axes[0, 1].set_title('R²')
        axes[0, 1].set_xlabel('轮次')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # 绘制RMSE图
        train_rmse = [metrics['RMSE'] for metrics in train_metrics_history]
        val_rmse = [metrics['RMSE'] for metrics in val_metrics_history]
        axes[1, 0].plot(epochs, train_rmse, 'b-', label='训练 RMSE')
        axes[1, 0].plot(epochs, val_rmse, 'r-', label='验证 RMSE')
        axes[1, 0].set_title('RMSE')
        axes[1, 0].set_xlabel('轮次')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 绘制MAE图
        train_mae = [metrics['MAE'] for metrics in train_metrics_history]
        val_mae = [metrics['MAE'] for metrics in val_metrics_history]
        axes[1, 1].plot(epochs, train_mae, 'b-', label='训练 MAE')
        axes[1, 1].plot(epochs, val_mae, 'r-', label='验证 MAE')
        axes[1, 1].set_title('MAE')
        axes[1, 1].set_xlabel('轮次')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        # 设置整体标题
        if final:
            fig.suptitle(f'{self.dataset_name} - 最终训练结果 (共{len(epochs)}轮)', fontsize=16)
            save_suffix = 'final'
        else:
            fig.suptitle(f'{self.dataset_name} - 训练进度 (至第{len(epochs)}轮)', fontsize=16)
            save_suffix = f'epoch_{len(epochs)}'

        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # 选择保存目录
        if 'fold' in self.dataset_name:
            save_dir = Config.CV_OUTPUT_DIR
        else:
            save_dir = Config.TRAIN_OUTPUT_DIR

        # 保存图片
        save_path = os.path.join(save_dir, f'{self.dataset_name}_training_progress_{save_suffix}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        if final:
            print(f"最终训练进度图已保存到: {save_path}")

    def save_training_history(self, train_losses, val_losses, train_metrics_history, val_metrics_history):
        """保存训练历史"""
        history = {
            'epoch': range(1, len(train_losses) + 1),
            'train_mse_loss': train_losses,
            'val_mse_loss': val_losses
        }

        # 添加指标历史
        for metric in ['R²', 'RMSE', 'MAE']:
            history[f'train_{metric}'] = [metrics[metric] for metrics in train_metrics_history]
            history[f'val_{metric}'] = [metrics[metric] for metrics in val_metrics_history]

        # 保存为CSV
        import pandas as pd
        history_df = pd.DataFrame(history)

        # 判断是否是交叉验证的训练历史
        if 'fold' in self.dataset_name:
            save_path = os.path.join(Config.CV_OUTPUT_DIR, f"{self.dataset_name}_training_history.csv")
        else:
            save_path = os.path.join(Config.TRAIN_OUTPUT_DIR, f"{self.dataset_name}_training_history.csv")

        history_df.to_csv(save_path, index=False)
        print(f"训练历史已保存到: {save_path}")

    def predict(self, data_loader):
        """
        使用模型进行预测

        Args:
            data_loader: 数据加载器

        Returns:
            predictions: 预测结果
            actuals: 实际值
        """
        self.model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for features, target in data_loader:
                features, target = features.to(self.device), target.to(self.device)
                output = self.model(features)

                predictions.extend(output.cpu().numpy().flatten())
                actuals.extend(target.cpu().numpy().flatten())

        return np.array(predictions), np.array(actuals)
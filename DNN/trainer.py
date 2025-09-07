import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from config import Config


class PrecipitationModelTrainer:
    def __init__(self, model, dataset_name, device, learning_rate=None, weight_decay=None):
        print("初始化模型训练器...")
        self.model = model
        self.dataset_name = dataset_name
        self.device = device
        self.model.to(device)

        # 使用传入的学习率和权重衰减，如果未提供则使用配置中的默认值
        self.learning_rate = learning_rate or Config.LEARNING_RATE
        self.weight_decay = weight_decay or Config.WEIGHT_DECAY

        print(f"使用学习率: {self.learning_rate}, 权重衰减: {self.weight_decay}")

        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )

        # 使用余弦退火学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # 减少重启周期
            T_mult=1,  # 保持周期不变
            eta_min=1e-5  # 降低最小学习率
        )

        # 用于追踪最佳验证损失
        self.best_val_loss = float('inf')

        # 梯度裁剪阈值
        self.grad_clip_val = 0.5  # 降低梯度裁剪阈值

    def train(self, train_loader, val_loader, epochs=200):
        print("开始训练...")
        train_losses = []
        val_losses = []
        train_metrics_history = []
        val_metrics_history = []
        best_val_loss = float('inf')
        best_model_state = None
        no_improve_epochs = 0

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_predictions = []
            train_targets = []
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')

            for features, targets in train_pbar:
                features, targets = features.to(self.device), targets.to(self.device)
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()

                # 应用梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_val)

                self.optimizer.step()

                train_loss += loss.item()
                train_predictions.extend(outputs.detach().cpu().numpy())
                train_targets.extend(targets.cpu().numpy())
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # 更新学习率
            self.scheduler.step()

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
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

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

            current_lr = self.optimizer.param_groups[0]['lr']

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                no_improve_epochs = 0
                # 保存最佳模型
                self.save_model(best_model_state, epoch)
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= Config.PATIENCE:
                print(f'早停: {epoch + 1} 轮后验证损失未改善')
                break

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'\nEpoch [{epoch + 1}/{epochs}]')
                print(f'学习率: {current_lr:.6f}')
                print('训练集:')
                print(f'  损失: {train_loss:.4f}')
                for metric, value in train_metrics.items():
                    print(f'  {metric}: {value:.4f}')
                print('验证集:')
                print(f'  损失: {val_loss:.4f}')
                for metric, value in val_metrics.items():
                    print(f'  {metric}: {value:.4f}')

        self.model.load_state_dict(best_model_state)
        print(f"\n训练结束！最佳验证损失: {best_val_loss:.4f}")

        # 保存训练历史
        self.save_training_history(train_losses, val_losses, train_metrics_history, val_metrics_history)

        return train_losses, val_losses

    def save_model(self, model_state, epoch):
        """保存模型"""
        save_path = os.path.join(Config.TRAIN_OUTPUT_DIR, f"{self.dataset_name}_model.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay
        }, save_path)
        print(f"模型已保存到: {save_path}")

    def save_training_history(self, train_losses, val_losses, train_metrics_history, val_metrics_history):
        """保存训练历史"""
        history = {
            'epoch': range(1, len(train_losses) + 1),
            'train_loss': train_losses,
            'val_loss': val_losses
        }

        # 添加指标历史
        for metric in ['R²', 'RMSE', 'MAE']:
            history[f'train_{metric}'] = [metrics[metric] for metrics in train_metrics_history]
            history[f'val_{metric}'] = [metrics[metric] for metrics in val_metrics_history]

        # 保存为CSV
        import pandas as pd
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(Config.TRAIN_OUTPUT_DIR, f"{self.dataset_name}_training_history.csv"),
                          index=False)
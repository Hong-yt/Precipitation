import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config


def modify_and_visualize_shap():
    """
    修改SHAP可视化结果，删除特定特征并重命名其他特征
    """
    try:
        # 设置字体
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.unicode_minus'] = False

        # 读取SHAP值和特征数据
        shap_file = os.path.join(Config.SHAP_OUTPUT_DIR, f"{Config.DATASET_NAME}_overall_shap_values.npy")
        features_file = os.path.join(Config.SHAP_OUTPUT_DIR, f"{Config.DATASET_NAME}_overall_features.npy")

        if not os.path.exists(shap_file) or not os.path.exists(features_file):
            print("错误：找不到SHAP值或特征数据文件")
            return

        shap_values = np.load(shap_file)
        features = np.load(features_file)

        # 获取原始特征名称
        feature_names = [
            'tc_r', 'cdr_eivd_r', 'chirps_eivd_r', 'cmo_eivd_r', 'mvkg_eivd_r',
            'nrt_eivd_r', 'nrtg_eivd_r', 'ERA5_R','SM2RAIN_R','AI','LAI','DEM','landuse','SAND','SILT','CLAY',
            'AI_rep', 'LAI_rep', 'DEM_rep', 'landuse_rep', 'SAND_rep', 'SILT_rep', 'CLAY_rep'
        ]

        # 过滤掉不需要的特征
        exclude_features = ['cmo_eivd_r', 'mvkg_eivd_r', 'nrt_eivd_r', 'nrtg_eivd_r', 'chirps_eivd_r']
        keep_indices = [i for i, name in enumerate(feature_names) if name not in exclude_features]

        # 更新特征名称和SHAP值
        feature_names = [feature_names[i] for i in keep_indices]
        shap_values = shap_values[:, keep_indices]
        features = features[:, keep_indices]

        # 替换特征名称
        feature_names = ['R(IMERG-SM2RAIN)' if name == 'SM2RAIN_R' else 'R(IMERG-ERA5)' if name == 'ERA5_R' else 'TC-R' if name == 'tc_r' else 'EIVD-R' if name == 'cdr_eivd_r' else name
                         for name in feature_names]

        # 创建带有两个子图的组合图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10), gridspec_kw={'width_ratios': [1, 2]})

        # ===== 左侧：平均SHAP值条形图 =====
        # 计算特征重要性
        mean_abs_shap = np.abs(shap_values).mean(0)
        sorted_idx = np.argsort(-mean_abs_shap)
        top_mean_abs_shap = mean_abs_shap[sorted_idx]
        top_names = [feature_names[i] for i in sorted_idx]

        # 为条形图创建浅蓝色背景
        for i in range(len(top_names)):
            ax1.axhspan(i - 0.4, i + 0.4, color='#E6F3FF', alpha=0.6)

        # 绘制水平条形图
        bars = ax1.barh(range(len(top_names)), top_mean_abs_shap, color='#4292C6', height=0.7, alpha=0.8)

        # 设置Y轴刻度（特征名称）和标签
        ax1.set_yticks(range(len(top_names)))
        ax1.set_yticklabels(top_names, fontsize=14)
        ax1.invert_yaxis()  # 最大值在顶部

        # 设置网格线
        ax1.grid(axis='x', linestyle='--', alpha=0.3)

        # 设置X轴范围从0开始
        ax1.set_xlim(0, max(top_mean_abs_shap) * 1.1)
        ax1.tick_params(axis='x', labelsize=14)

        # 添加标题
        ax1.set_title('Mean Shapley Value (Feature Importance)', fontsize=19, pad=10, fontfamily='Times New Roman')

        # ===== 右侧：蜂群图 =====
        # 使用更清晰的颜色映射
        cmap = plt.cm.coolwarm

        # 为所有特征创建蜂群图
        for i, idx in enumerate(sorted_idx):
            # 获取当前特征的SHAP值和特征值
            feature_shap_values = shap_values[:, idx]
            feature_values = features[:, idx]

            # 过滤掉NaN或inf值
            valid_indices = np.isfinite(feature_shap_values) & np.isfinite(feature_values)
            feature_shap_values = feature_shap_values[valid_indices]
            feature_values = feature_values[valid_indices]

            if len(feature_values) == 0:
                continue

            # 为蜂群图创建浅蓝色背景
            ax2.axhspan(i - 0.4, i + 0.4, color='#E6F3FF', alpha=0.6)

            # 计算每个点的垂直抖动
            y_jitter = np.random.normal(scale=0.1, size=len(feature_shap_values))

            # 绘制散点图
            sc = ax2.scatter(
                feature_shap_values,
                np.ones(len(feature_shap_values)) * i + y_jitter,
                c=feature_values,
                cmap=cmap,
                s=25,
                alpha=0.75,
                edgecolor='none'
            )

        # 添加垂直线表示SHAP值为0
        ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.5)

        # 设置Y轴刻度和标签
        ax2.set_yticks(range(len(top_names)))
        ax2.set_yticklabels([''] * len(top_names))  # 空标签，因为左侧已经有了
        ax2.invert_yaxis()  # 最大值在顶部

        # 设置网格线
        ax2.grid(axis='x', linestyle='--', alpha=0.3)

        # 设置X轴标签
        ax2.set_title('Shapley Value Contribution', fontsize=19, pad=10, fontfamily='Times New Roman')
        ax2.tick_params(axis='x', labelsize=14)

        # 计算X轴范围使其更紧凑
        max_shap = np.max(np.abs(shap_values))
        ax2.set_xlim(-max_shap * 1.05, max_shap * 1.05)

        # 添加颜色条
        cbar = plt.colorbar(sc, ax=ax2, pad=0.02, aspect=30)
        cbar.set_ticks([])
        cbar.ax.text(0.5, -0.01, 'Low', ha='center', va='top', fontsize=13,
                     transform=cbar.ax.transAxes)
        cbar.ax.text(0.5, 1.01, 'High', ha='center', va='bottom', fontsize=13,
                     transform=cbar.ax.transAxes)
        cbar.set_label('Feature Value', fontsize=15, rotation=270, labelpad=15)

        # 设置整体标题
        plt.suptitle(f'SHAP Summary: {Config.DATASET_NAME}', fontsize=24, y=0.98, fontfamily='Times New Roman')

        # 优化布局
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # 保存图片
        output_file = os.path.join(Config.SHAP_OUTPUT_DIR, f"{Config.DATASET_NAME}_modified_shap_combined.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"修改后的SHAP组合图已保存至 {output_file}")

    except Exception as e:
        print(f"修改SHAP可视化时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    modify_and_visualize_shap()
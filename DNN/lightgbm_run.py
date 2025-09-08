import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from scipy.stats import pearsonr
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
# import xgboost as xgb
import random
import os
from scipy.stats import boxcox
import shap
import joblib

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


# ASCAT_ISMN_TCa_R_ECC = pd.read_csv(r'H:\global_soil_moisture_error_analysis\Inconsistent_experiment_20240318\data_surface\ISMN\1008sensor\Final\ASCAT.csv')
# ERA5_ISMN_TCa_R_ECC = pd.read_csv(r'H:\global_soil_moisture_error_analysis\Inconsistent_experiment_20240318\data_surface\ISMN\1008sensor\Final\ERA5.csv')
# SMAPL3_ISMN_TCa_R_ECC = pd.read_csv(r'H:\global_soil_moisture_error_analysis\Inconsistent_experiment_20240318\data_surface\ISMN\1008sensor\Final\SMAPL3.csv')
#
#
# # merge = ASCAT_ISMN_TCa_R_ECC.dropna()
# merge = SMAPL3_ISMN_TCa_R_ECC.dropna()
# # merge = ERA5_ISMN_TCa_R_ECC.dropna()
#
#
# # 将特征和目标拆分
# # X = merge[['ASCAT_rho', 'ASCAT_stderr', 'tag_percent', 'ECC_ascat_smap', 'ECC_era5_ascat', 'ASCAT_SMAPL3_rho',
# #            'ERA5_ASCAT_rho', 'precipitation', 'soiltemperature', 'DEM', 'dem_percent', 'AI', 'AI_percent', 'LAI',
# #            'LAI_percent', 'landcover', 'landcover_percent', 'SAND', 'sand_percent', 'CLAY', 'clay_percent', 'SILT',
# #            'silt_percent']]
#
# X = merge[['SMAPL3_rho', 'SMAPL3_stderr', 'tag_percent', 'ECC_ascat_smap', 'ECC_smap_era5', 'ASCAT_SMAPL3_rho', 'SMAPL3_ERA5_rho', 'precipitation', 'soiltemperature', 'DEM', 'dem_percent', 'AI', 'AI_percent', 'LAI', 'LAI_percent', 'landcover', 'landcover_percent', 'SAND', 'sand_percent', 'CLAY', 'clay_percent', 'SILT', 'silt_percent']]
# # X = merge[['ERA5_rho', 'ERA5_stderr', 'tag_percent', 'ECC_smap_era5', 'ECC_era5_ascat', 'SMAPL3_ERA5_rho', 'ERA5_ASCAT_rho', 'precipitation', 'soiltemperature', 'DEM', 'dem_percent', 'AI', 'AI_percent', 'LAI', 'LAI_percent', 'landcover', 'landcover_percent', 'SAND', 'sand_percent', 'CLAY', 'clay_percent', 'SILT', 'silt_percent']]
#
# y = merge['rho']
#
# # 创建LGBM回归模型
# params = {
#     'boosting_type': 'gbdt',
#     'num_leaves': 127,
#     'learning_rate': 0.05,
#     'feature_fraction': 0.8,
#     'bagging_fraction': 0.9,
#     'bagging_freq': 5,
#     'lambda_l1': 0.05,
#     'lambda_l2': 0.04,
#     'min_child_samples': 20,
#     'max_depth': 11,
#     'n_estimators': 170
# }
#
#
# y1 = []
# y2 = []
# TC = []
# num_folds = 5  # 折数
# kf = KFold(n_splits=num_folds, shuffle=True, random_state=131)  # 创建交叉验证对象
# mse_scores = []
# i = 0
# accumulated_df = pd.DataFrame()
# for train_index, val_index in kf.split(X):
#     i += 1
#     X_train, X_val = X.iloc[train_index], X.iloc[val_index]
#     y_train, y_val = y.iloc[train_index], y.iloc[val_index]
#     # TCR = X_val['SMAPL3_rho']
#
#     model = lgb.LGBMRegressor(**params)
#     # model = RandomForestRegressor()
#     # model = RandomForestRegressor(bootstrap=True, criterion='squared_error', max_depth=None, max_features='auto',
#     #                               min_samples_leaf=1,
#     #                               min_samples_split=2, n_estimators=150, oob_score=False, random_state=42)
#     # model = xgb.XGBRegressor()
#     model.fit(X_train, y_train)
#
#     y_pred = model.predict(X_val)
#     # joblib.dump(model, f'../model/ERA5_model{i}.pkl')
#     y1.append(y_val)
#     y2.append(y_pred)
#     # TC.append(TCR)
#
#
#
#     # feature_importance = model.feature_importances_
#     # feature_names = X.columns
#     #
#     # # 对特征重要性进行排序
#     # sorted_indices = np.argsort(feature_importance)
#     # sorted_feature_importance = feature_importance[sorted_indices]
#     # sorted_feature_names = feature_names[sorted_indices]
#     #
#     # plt.figure(figsize=(10, 6))
#     # plt.barh(range(len(sorted_feature_importance)), sorted_feature_importance, align='center')
#     # plt.yticks(range(len(sorted_feature_importance)), sorted_feature_names)
#     # plt.xlabel('Feature Importance')
#     # plt.ylabel('Features')
#     # plt.title('Sorted Feature Importance')
#     # plt.show()
#
#     # correlation, _ = pearsonr(y_pred, y_val)
#     # print("r:", correlation)
#     # print("MAE:", mean_absolute_error(y_pred, y_val))
#     # print("MSE:", mean_squared_error(y_pred, y_val))
#     # print("RMSE:", sqrt(mean_squared_error(y_pred, y_val)))
#     # print("r2:", r2_score(y_pred, y_val))
#     # print('all:', len(y1))
#     # r2 = model.score(X_val, y_val)
#     # print(r2)
#
# y1 = np.concatenate(y1)
# y2 = np.concatenate(y2)
# # TC = np.concatenate(TC)
# # df = pd.DataFrame({'y_true': y1, 'y_pred': y2, 'TCR': TC})
# # df.to_csv('H:\\global_soil_moisture_error_analysis\\Inconsistent_experiment_20240318\\data_surface\\ISMN\\1008sensor\\Final\\figure\\origin_figure_data\\SMAPL3_true_pred_TCR.csv', index=False)
# correlation, _ = pearsonr(y1, y2)
# print("r:", correlation)
# print("MAE:", mean_absolute_error(y1, y2))
# print("MSE:", mean_squared_error(y1, y2))
# print("RMSE:", sqrt(mean_squared_error(y1, y2)))
# print("r2:", r2_score(y1, y2))
# print('all:', len(y1))

features = pd.read_excel('D:/hongyouting/data/dnn/merge/features/imerg_f7_features.xlsx')
target = pd.read_excel('D:/hongyouting/data/dnn/merge/target/imerg_f7_target.xlsx')

X = features.iloc[:,2:]
y = target.iloc[:,2:]

# 打印数据基本信息，帮助调试
print("特征数据形状:", X.shape)
print("目标变量形状:", y.shape)
print("特征包括:", X.columns.tolist())

# 可选：检查数据是否有缺失值并处理
# print("特征数据中的缺失值:", X.isnull().sum().sum())
# print("目标变量中的缺失值:", y.isnull().sum())

# 如果有缺失值，可以选择删除包含缺失值的行
# X_clean = X.dropna()
# y_clean = y[X.notna().all(axis=1)]

# 或者可以选择填充缺失值
# X = X.fillna(X.mean())  # 用均值填充

# 确保y是一维数组
y = y.values.ravel()

# # 创建LGBM回归模型
# params = {
#     'boosting_type': 'gbdt',
#     'num_leaves': 127,
#     'learning_rate': 0.05,
#     'feature_fraction': 0.8,
#     'bagging_fraction': 1,
#     'bagging_freq': 5,
#     'lambda_l1': 0.05,
#     'lambda_l2': 0.04,
#     'min_child_samples': 20,
#     'max_depth': 15,
#     'n_estimators': 161
# }
# params = {
#     'boosting_type': 'gbdt',
#     'num_leaves': 127,
#     'learning_rate': 0.05,
#     'colsample_bytree': 0.8,  # 替代 feature_fraction
#     'subsample': 1,  # 替代 bagging_fraction
#     'subsample_freq': 5,  # 替代 bagging_freq
#     'reg_alpha': 0.05,  # 替代 lambda_l1
#     'reg_lambda': 0.04,  # 替代 lambda_l2
#     'min_child_samples': 20,
#     'max_depth': 15,
#     'n_estimators': 161,
#     'min_child_weight': 1,
#     'min_split_gain': 0.0,
#     'verbose': -1,  # 减少警告信息
#     'force_col_wise': True  # 强制使用列并行
# }
# 定义参数网格
param_grid = {
    'num_leaves': [63, 127],
    'learning_rate': [0.05, 0.01],
    'n_estimators': [150, 200, 300],
    'max_depth': [8, 15],
    'min_child_samples': [10, 20],
    'reg_alpha': [0.03, 0.05],
    'reg_lambda': [0.03, 0.05],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
# 创建基础模型
base_model = lgb.LGBMRegressor(
    boosting_type='gbdt',
    objective='regression',
    metric='rmse',
    verbose=-1,
    force_col_wise=True,
    random_state=42
)

# 创建网格搜索对象
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)
print("开始网格搜索...")
# 执行网格搜索
grid_search.fit(X, y)

# 输出最佳参数
print("\n最佳参数:")
for param, value in grid_search.best_params_.items():
    print(f"{param}: {value}")

print(f"\n最佳交叉验证分数: {-grid_search.best_score_:.4f}")

# 使用最佳参数创建最终模型
best_params = grid_search.best_params_
best_params.update({
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'verbose': -1,
    'force_col_wise': True,
    'random_state': 42
})

# 初始化SHAP值累加器
shap_value_all = 0

# 存储实际值和预测值的列表
y_true_list = []
y_pred_list = []
shapey = []

# 使用最佳参数进行交叉验证
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=80)
# mse_scores = []

# accumulated_df = pd.DataFrame()
# 执行交叉验证
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"\n执行第 {fold + 1} 折交叉验证...")

    # 划分训练集和验证集
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    # y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # 创建并训练模型
    model = lgb.LGBMRegressor(**best_params)  # LightGBM
    # model = RandomForestRegressor(n_estimators=150, random_state=42)  # 随机森林
    # model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, random_state=42)  # XGBoost

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    # 存储实际值和预测值
    y_true_list.append(y_val)
    y_pred_list.append(y_pred)

    # feature_importance = model.feature_importances_
    # feature_names = X.columns
    # #对特征重要性进行排序
    # sorted_indices = np.argsort(feature_importance)
    # sorted_feature_importance = feature_importance[sorted_indices]
    # sorted_feature_names = feature_names[sorted_indices]
    # plt.figure(figsize=(10, 6))
    # plt.barh(range(len(sorted_feature_importance)), sorted_feature_importance, align='center')
    # plt.yticks(range(len(sorted_feature_importance)), sorted_feature_names)
    # plt.xlabel('Feature Importance')
    # plt.ylabel('Features')
    # plt.title('Sorted Feature Importance')
    # plt.show()


    explainer = shap.TreeExplainer(model)
    # 以numpy数组的形式输出SHAP值
    shap_values = explainer.shap_values(X)
    # shap.summary_plot(shap_values, X_train, plot_type="bar")
    shap_value_all = shap_value_all+shap_values
    shapey.append(shap_values)
    # print(shap_values)


    # correlation, _ = pearsonr(y_pred, y_val)
    # print("r:", correlation)
    # print("MAE:", mean_absolute_error(y_pred, y_val))
    # print("MSE:", mean_squared_error(y_pred, y_val))
    # print("RMSE:", sqrt(mean_squared_error(y_pred, y_val)))
    # print("r2:", r2_score(y_pred, y_val))
    # print('all:', len(y1))
    # r2 = model.score(X_val, y_val)
    # print(r2)

    # 输出每折的性能指标
    fold_corr, _ = pearsonr(y_val, y_pred)
    fold_mae = mean_absolute_error(y_val, y_pred)
    fold_rmse = sqrt(mean_squared_error(y_val, y_pred))
    fold_r2 = r2_score(y_val, y_pred)

    print(f"第 {fold + 1} 折结果: 相关系数={fold_corr:.4f}, MAE={fold_mae:.4f}, RMSE={fold_rmse:.4f}, R²={fold_r2:.4f}")

# 合并所有折的结果
y_true = np.concatenate(y_true_list)
y_pred = np.concatenate(y_pred_list)
# shape_v = np.concatenate(shapey)

# df = pd.DataFrame({'y_true': y1, 'y_pred': y2})
# df.to_csv('H:\\global_soil_moisture_error_analysis\\Inconsistent_experiment_20240318\\data_surface\\ISMN\\1008sensor\\Final\\figure\\origin_figure_data\\SMAPL3_true_pred.csv', index=False)
correlation, _ = pearsonr(y_true, y_pred)
print("\n整体性能指标:")
print("r:", correlation)
print("MAE:", mean_absolute_error(y_true, y_pred))
print("MSE:", mean_squared_error(y_true, y_pred))
print("RMSE:", sqrt(mean_squared_error(y_true, y_pred)))
print("r2:", r2_score(y_true, y_pred))
print('all:', len(y_true))
print("bias:", np.mean(np.abs(y_true - y_pred)))

# shape_v = shap_value_all/5
# # # shap.summary_plot(shape_v, X)
# # # shap.summary_plot(shape_v, X, plot_type="bar")
# shap.dependence_plot('ASCAT_SMAPL3_rho', shape_v, X, interaction_index=None)
# # shap.plots.partial_dependence('ASCAT_SMAPL3_rho', shape_v, X, ice=False,model_expected_value=True, feature_expected_value=True)
# #
# abs_arr = np.abs(shape_v)
# abs_means = np.mean(abs_arr, axis=0)
#
# feature_importance = abs_means
# feature_names = X.columns
#
# #对特征重要性进行排序
# sorted_indices = np.argsort(feature_importance)
# sorted_feature_importance = feature_importance[sorted_indices]
# sorted_feature_names = feature_names[sorted_indices]
# plt.figure(figsize=(10, 6))
# plt.barh(range(len(sorted_feature_importance)), sorted_feature_importance, align='center')
# plt.yticks(range(len(sorted_feature_importance)), sorted_feature_names)
# plt.xlabel('Feature Importance')
# plt.ylabel('Features')
# plt.title('Sorted Feature Importance')
# plt.show()
# #
# # data = pd.DataFrame()
# # for i in range(len(sorted_feature_importance)):
# #     name1= sorted_feature_names[i]
# #     importance = sorted_feature_importance[i]
# #     data.loc[name1, 0] = importance
# # data.to_csv('H:\\global_soil_moisture_error_analysis\\Inconsistent_experiment_20240318\\data_surface\\ISMN\\1008sensor\\Final\\figure\\origin_figure_data\\PDP\\ERA5_shap_importance_mean.csv', index=True)
# # print()
# #
# sorted_shape_v = shape_v[:, sorted_indices]
# data = pd.DataFrame()
# for i in range(len(sorted_feature_importance)):
#     name1= sorted_feature_names[i]
#     importance = sorted_shape_v[:, i]
#     data[name1] = importance
# #
# # data.to_csv('H:\\global_soil_moisture_error_analysis\\Inconsistent_experiment_20240318\\data_surface\\ISMN\\1008sensor\\Final\\figure\\origin_figure_data\\PDP\\ERA5_shap_importance.csv', index=False)
# # print()


# import seaborn as sns
# from scipy.optimize import fsolve
# # 绘制散点图
# plt.figure(figsize=(8, 6), dpi=300)
# plt.scatter(X['ASCAT_rho'], data['ASCAT_rho'], s=20, label='SHAP values', alpha=0.7)
# sns.regplot(x=X['ASCAT_rho'], y=data['ASCAT_rho'], scatter=False, lowess=True, color='lightcoral', label='LOWESS Curve')
# lowess_data = sns.regplot(x=X['ASCAT_rho'], y=data['ASCAT_rho'], scatter=False, lowess=True, color='lightcoral')
# line = lowess_data.get_lines()[0]  # 拟合线条对象
# x_fit = line.get_xdata()  # LOWESS 拟合线的 x 轴数据
# y_fit = line.get_ydata()  # LOWESS 拟合线的 y 轴数据
#
# plt.axhline(y=0, color='black', linestyle='-.', linewidth=1, label='SHAP = 0')
# plt.legend()
# plt.xlabel('Age', fontsize=12)
# plt.ylabel('SHAP value for\nAge', fontsize=12)
# ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# # plt.savefig("SHAP Dependence Plot_with_Multiple_Intersections.pdf", format='pdf', bbox_inches='tight')
# plt.show()
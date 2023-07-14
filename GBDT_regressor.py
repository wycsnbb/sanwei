import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# 读取数据
data = pd.read_excel('per_gdp.xlsx')
# 指定默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# 提取特征变量和标签变量
X = data.drop(columns=['大洲', '国家', '样本', '人均GDP（美元）', '经济（美元）', '人口（人）', '年份', '汇率', '原币种'])
y = data['人均GDP（美元）']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------ 首先采用GBDT的方法完成特征重要度的排序 ---------------------------------------------
# 但是GBDT方法存在一个非常大的问题在于只能确定重要度的排序,而不能看出正相关或者负相关
names = X.columns
estimator_GBDT = GradientBoostingRegressor()
estimator_GBDT.fit(X_train, y_train)

# 这里存放两个空列表,用于存放GBDT的特征和重要性
feature_importance = sorted(zip(estimator_GBDT.feature_importances_, names), reverse=True)
GBDT_fea = []
GBDT_import = []
for importance, feature in feature_importance:
    GBDT_fea.append(feature)
    GBDT_import.append(importance)

# ------------------------------ 为了看出特征向量与标签向量的正负相关性,采用线性回归来观测特征的重要性 -------------------------------
estimator_linear = LinearRegression()
estimator_linear.fit(X_train, y_train)

# 这里存放两个空列表,用于存放线性回归的特征和重要性
feature_cor = sorted(zip(estimator_linear.coef_, names), key=lambda x: abs(x[0]), reverse=True)
Lin_fea = []
Lin_import = []
for value, attr in feature_cor:
    Lin_fea.append(attr)
    Lin_import.append(value)

# ----------------------------------------------- 这里将具体数值传入excel文件中 ---------------------------------------
# 将特征的重要性写入excel表格中
res_GBDT_df = pd.DataFrame()
res_GBDT_df['特征名称'] = GBDT_fea
res_GBDT_df['特征重要性'] = GBDT_import
res_GBDT_df.to_excel('GBDT特征重要性.xlsx')

# 将线性回归的特征重要性写入excel表格
res_Lin_df = pd.DataFrame()
res_Lin_df['特征名称'] = Lin_fea
res_Lin_df['特征重要性'] = Lin_import
res_Lin_df.to_excel('线性回归特征重要性.xlsx')

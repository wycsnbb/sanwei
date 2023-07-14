import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from scipy.stats import spearmanr

image_path = 'economic_result'


# 所有特征向量组合起来的字典
CLASSES = {'类别flat_roof': 0, '类别gable_roof': 0, '类别gambrel_roof': 0, '类别row_roof': 0, '类别multiple_eave_roof': 0,
           '类别hipped_roof_v1': 0, '类别hipped_roof_v2': 0, '类别mansard_roof': 0, '类别pyramid_roof': 0, '类别arched_roof': 0,
           '类别revolved': 0, '类别other': 0, 'flat_roof面积': 0, 'gable_roof面积': 0, 'gambrel_roof面积': 0, 'row_roof面积': 0,
           'multiple_eave_roof面积': 0, 'hipped_roof_v1面积': 0, 'hipped_roof_v2面积': 0, 'mansard_roof面积': 0,
           'pyramid_roof面积': 0, 'arched_roof面积': 0, 'revolved面积': 0, 'other面积': 0, 'flat_roof密度': 0, 'gable_roof密度': 0,
           'gambrel_roof密度': 0, 'row_roof密度': 0, 'multiple_eave_roof密度': 0, 'hipped_roof_v1密度': 0,
           'hipped_roof_v2密度': 0, 'mansard_roof密度': 0, 'pyramid_roof密度': 0, 'arched_roof密度': 0, 'revolved密度': 0,
           'other密度': 0, 'flat_roof占比': 0, 'gable_roof占比': 0, 'gambrel_roof占比': 0, 'row_roof占比': 0,
           'multiple_eave_roof占比': 0, 'hipped_roof_v1占比': 0, 'hipped_roof_v2占比': 0, 'mansard_roof占比': 0,
           'pyramid_roof占比': 0, 'arched_roof占比': 0, 'revolved占比': 0, 'other占比': 0, 'flat_roof比例': 0, 'gable_roof比例': 0,
           'gambrel_roof比例': 0, 'row_roof比例': 0, 'multiple_eave_roof比例': 0, 'hipped_roof_v1比例': 0,
           'hipped_roof_v2比例': 0, 'mansard_roof比例': 0, 'pyramid_roof比例': 0, 'arched_roof比例': 0, 'revolved比例': 0,
           'other比例': 0, 'Low_rise': 0, 'Multi_storey': 0, 'Medium_and_high_rise': 0, 'High_rise_v1': 0,
           'High_rise_v2': 0, 'total_number': 0, 'total_area': 0}

# 用于显示的字典
LABELS_1 = ["flat", "gable", "gambrel", "row", "multi_eave", "hipped_v1", "hipped_v2", "mansard", "pyramid",
            "arched", "revolved", "other"]
LABELS_2 = ["Low", "Multi_storey", "Med_high", "High_v1", "High_v2"]
LABELS_3 = ["Total_Num", "Total_Area"]

# 创建一个空的DataFrame
spear_cor_df = pd.DataFrame()

country_result = []

# 读取excel文件
data = pd.read_excel('per_gdp.xlsx')

# 设置标签
label = '人均GDP（美元）'
target = np.array(data[label])

# 初始化各种相关性系数
corr_sum = {}
spear_cor_sum = {}
kendall_cor_sum = {}

for item in CLASSES:
    variable = np.array(data[item])
    new_variable = variable
    new_variable[new_variable > 10] = np.log(new_variable[new_variable > 10])
    # 首先计算皮尔森相关系数
    cor = np.corrcoef(target, new_variable)[0, 1]
    corr_sum[item] = cor

    # 其次计算斯皮尔曼相关系数
    spear_cor, p_value = spearmanr(target, variable)
    spear_cor_sum[item] = spear_cor

    # 最后计算肯德尔相关系数
    kendall_cor, kendall_p_value = kendalltau(target, variable)
    kendall_cor_sum[item] = kendall_cor
    # CLASSES[item] = kendall_cor

keys_1 = list(LABELS_1)
keys_2 = list(LABELS_2)
keys_3 = list(LABELS_3)
values = list(spear_cor_sum.values())

# 添加当前城市的相关系数到DataFrame
df = pd.DataFrame.from_dict(CLASSES, orient='index')
country_result.append(df)

# 对相关性分析结果作图
slices = [slice(0, 12), slice(12, 24), slice(24, 36), slice(36, 48), slice(48, 60), slice(60, 65), slice(65, 67)]
slice_names = ["细粒度数量与人均GRP相关性", "细粒度面积与人均GRP的相关性", "细粒度密度与人均GRP相关性", "细粒度占比与人均GRP的相关性",
               "细粒度比例与人均GRP的相关性", "建筑物高度与人均GRP的相关性", "建筑物面积数量与人均GRP的相关性"]

for idx, s in enumerate(slices):

    plt.figure(figsize=(12, 8))
    # 设置中文字体
    plt.rcParams["font.family"] = "SimHei"
    # 解决保存图像是负号'-'显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False
    values_item = values[s]
    if idx < 5:
        plt.bar(keys_1, values_item)
        # 添加文本
        for x, y in zip(keys_1, values_item):
            plt.text(x, y + 0.01, '%.2f' % y, ha='center')
        plt.xlabel("细粒度")

    elif idx == 5:
        plt.bar(keys_2, values_item)
        # 添加文本
        for x, y in zip(keys_2, values_item):
            plt.text(x, y + 0.01, '%.2f' % y, ha='center')
        plt.xlabel("建筑物高度")

    else:
        plt.bar(keys_3, values_item)
        # 添加文本
        for x, y in zip(keys_3, values_item):
            plt.text(x, y + 0.01, '%.2f' % y, ha='center')
        plt.xlabel("建筑物的数量和面积")

    plt.ylabel("Spearman correlation")
    title = f"{slice_names[idx]}"
    plt.title(title)
    # 保存当前柱状图
    plt.savefig(f'{image_path}/Spearman/{slice_names[idx]}.png')


# total_df = pd.concat(country_result, axis=0, ignore_index=True)
# total_df.to_excel('kendall_correlations.xlsx')




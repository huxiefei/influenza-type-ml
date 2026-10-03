# -*- coding: utf-8 -*-
"""
项目15: WBC相关性矩阵 + 分组SHAP重要性
回应 R4-8 / R5：白细胞总数与亚群绝对计数/百分比算术相关，SHAP 归因会分摊。
给出：① 相关矩阵热图；② 算术相关说明；③ 按分组（WBC 总数 / 绝对计数 / 百分比 / 红系 / 血小板）汇总 SHAP 贡献。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import lightgbm as lgb, shap

BASE = os.environ.get("INFLUENZA_DATA_DIR", r"E:\上班后的资料\甲乙流多中心3")
OUT  = os.path.join(BASE, r'代码存档\运行结果\15_WBC相关性')
os.makedirs(OUT, exist_ok=True)

ALL_NUM = ['age', 'WBC', 'NEU', 'NEU%', 'LYM', 'LYM%', 'BASO', 'BASO%',
           'EO', 'EO%', 'MON', 'MON%', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH',
           'MCHC', 'PLT', 'MPV', 'PCT', 'PDW', 'RDW-CV', 'month']

# 按生理意义分组
GROUPS = {
    'Demographics':  ['age', 'sex'],
    'Temporal':      ['month'],
    'WBC total':     ['WBC'],
    'WBC absolute':  ['NEU', 'LYM', 'MON', 'EO', 'BASO'],
    'WBC percent':   ['NEU%', 'LYM%', 'MON%', 'EO%', 'BASO%'],
    'RBC indices':   ['RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW-CV'],
    'Platelet':      ['PLT', 'MPV', 'PCT', 'PDW'],
}

TAH = pd.read_csv(os.path.join(BASE, 'a附三院数据_processed.csv'))
X = TAH[ALL_NUM + ['sex']]; y = TAH['RESULT'].values

# 1) 相关矩阵（仅数值特征）
num_cols = [c for c in ALL_NUM if c != 'sex']
corr = TAH[num_cols].corr()
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, square=True, cbar_kws={'shrink': 0.7})
plt.title('Spearman-style correlation among numerical features (TAH derivation)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()
corr.to_csv(os.path.join(OUT, 'correlation_matrix.csv'), encoding='utf-8-sig')

# 强相关对（|r|>0.8）
strong = []
for i in range(len(num_cols)):
    for j in range(i+1, len(num_cols)):
        r = corr.iloc[i,j]
        if abs(r) > 0.8:
            strong.append({'feat1': num_cols[i], 'feat2': num_cols[j], 'r': float(r)})
strong_df = pd.DataFrame(strong).sort_values('r', key=abs, ascending=False)
strong_df.to_csv(os.path.join(OUT, 'strong_correlations.csv'),
                  index=False, encoding='utf-8-sig')

# 2) 训练模型 + 计算SHAP
num_cols_full = ALL_NUM
pre = ColumnTransformer([
    ('num', StandardScaler(), [c for c in ALL_NUM if c != 'sex']),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), ['sex']),
], remainder='drop')
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.30,
                                       stratify=y, random_state=42)
Xt = pre.fit_transform(Xtr)
Xv = pre.transform(Xva)

model = lgb.LGBMClassifier(objective='multiclass', num_class=3,
                             learning_rate=0.05, n_estimators=100, num_leaves=31,
                             random_state=42, verbosity=-1)
model.fit(Xt, ytr)
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(Xv)  # list of (n, p) for each class
# process feature names (after ColumnTransformer)
proc_names = list(pre.get_feature_names_out())
# strip 'num__' and 'cat__'
clean_names = [n.replace('num__','').replace('cat__','') for n in proc_names]

# 3) 分组汇总 mean|SHAP| （三分类平均）
mean_abs = np.mean([np.abs(s) for s in sv], axis=(0,2))  # 平均所有样本×所有类别
order = np.argsort(mean_abs)[::-1]
# 按 GROUPS 汇总
group_importance = []
for g, members in GROUPS.items():
    members_idx = [i for i, n in enumerate(clean_names) if n in members]
    if not members_idx:
        continue
    s = float(mean_abs[members_idx].sum())
    group_importance.append({'Group': g, 'Members': len(members_idx),
                             'Grouped_meanAbsSHAP': s,
                             'Top_member': clean_names[members_idx[np.argmax(mean_abs[members_idx])]],
                             'Top_member_SHAP': float(mean_abs[members_idx].max())})
group_importance = sorted(group_importance, key=lambda d: -d['Grouped_meanAbsSHAP'])
df_group = pd.DataFrame(group_importance)
df_group.to_csv(os.path.join(OUT, 'grouped_feature_importance.csv'),
                  index=False, encoding='utf-8-sig')

# 画 grouped bar
plt.figure(figsize=(10, 6))
plt.barh([d['Group'] for d in group_importance][::-1],
         [d['Grouped_meanAbsSHAP'] for d in group_importance][::-1],
         color='#1f77b4')
plt.xlabel('Sum of mean |SHAP value| (all 3 classes)')
plt.title('Grouped SHAP feature importance (TAH)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'grouped_importance.png'), dpi=300, bbox_inches='tight')
plt.close()

pd.set_option('display.width', 200)
print(f'Strong correlations |r|>0.8: {len(strong_df)} pairs')
print(strong_df.head(15).round(3).to_string(index=False))
print('\n=== 分组 SHAP 重要性（按生理分组汇总） ===')
print(df_group.round(4).to_string(index=False))
print(f'\nsaved to {OUT}')
# -*- coding: utf-8 -*-
"""
项目12: 折内预处理 Pipeline 重跑
将 ColumnTransformer 嵌入 Pipeline，GridSearchCV 对 Pipeline 操作，确保超参搜索时
每一折的训练子集内部独立 fit 预处理，避免折间信息泄露（虽然原稿此处泄露只影响
超参选择而非最终测试集评估，但审稿人指出了该问题，应在回复中说明）。
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import lightgbm as lgb

BASE = os.environ.get("INFLUENZA_DATA_DIR", r"E:\上班后的资料\甲乙流多中心3")
OUT  = os.path.join(BASE, r'代码存档\运行结果\12_Pipeline重跑')
os.makedirs(OUT, exist_ok=True)

ALL_NUM = ['age', 'WBC', 'NEU', 'NEU%', 'LYM', 'LYM%', 'BASO', 'BASO%',
           'EO', 'EO%', 'MON', 'MON%', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH',
           'MCHC', 'PLT', 'MPV', 'PCT', 'PDW', 'RDW-CV', 'month']

def make_pipe():
    num_cols = [c for c in ALL_NUM if c != 'sex']
    pre = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), ['sex']),
    ], remainder='drop')
    model = lgb.LGBMClassifier(
        objective='multiclass', num_class=3, metric='multi_logloss',
        learning_rate=0.05, n_estimators=100, num_leaves=31,
        random_state=42, verbosity=-1,
    )
    return Pipeline([('pre', pre), ('clf', model)])

# 网格（与原稿 GridSearchCV 一致）
PARAM_GRID = {
    'clf__num_leaves': [15, 31, 63],
    'clf__learning_rate': [0.03, 0.05, 0.1],
    'clf__n_estimators': [50, 100, 150],
}

TAH = pd.read_csv(os.path.join(BASE, 'a附三院数据_processed.csv'))
X = TAH[ALL_NUM + ['sex']]
y = TAH['RESULT'].values

# 70/30 split 保持与原稿一致（random_state=42）
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.30,
                                       stratify=y, random_state=42)

pipe = make_pipe()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipe, PARAM_GRID, scoring='roc_auc_ovr', cv=cv,
    n_jobs=-1, refit=True, return_train_score=False,
)
print('GridSearchCV on Pipeline (5-fold)...')
grid.fit(Xtr, ytr)
print(f'Best CV macro-AUC: {grid.best_score_:.4f}')
print(f'Best params: {grid.best_params_}')

# 评估验证集与外部三中心
best = grid.best_estimator_
pv = best.predict_proba(Xva)
va_macro = roc_auc_score(yva, pv, multi_class='ovr', average='macro')
print(f'Internal validation macro-AUC: {va_macro:.4f}')

rows = []
for name, fp in [('SAH','b新桥数据_processed.csv'),('CEMC','c急救数据.csv'),
                 ('FDH','d丰都数据_processed.csv')]:
    ext = pd.read_csv(os.path.join(BASE, fp))
    ye = ext['RESULT'].values
    Xe = ext[ALL_NUM + ['sex']]
    p = best.predict_proba(Xe)
    macro = roc_auc_score(ye, p, multi_class='ovr', average='macro')
    aA = roc_auc_score((ye==1).astype(int), p[:,1])
    aB = roc_auc_score((ye==2).astype(int), p[:,2])
    rows.append({'cohort': name, 'macro_auc': macro,
                 'FluA_auc': aA, 'FluB_auc': aB})
    print(f'External {name}: macro={macro:.4f}, FluA={aA:.4f}, FluB={aB:.4f}')

df = pd.DataFrame(rows)
df['internal_valid'] = va_macro
df['best_cv_score']  = grid.best_score_
df.to_csv(os.path.join(OUT, 'pipeline_metrics.csv'), index=False, encoding='utf-8-sig')
print(f'\nsaved to {OUT}')
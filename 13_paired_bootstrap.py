# -*- coding: utf-8 -*-
"""
项目13: 六模型配对bootstrap比较
基于07号脚本的predictions.csv 与02号脚本的相同拆分/GridSearchCV，
计算 LightGBM vs XGBoost vs RF 的配对 macro-AUC 差值及其95%CI（bootstrap 1000次）。
回应 R5："LightGBM... lack of formal statistical evidence...confidence intervals overlapped"
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import lightgbm as lgb, xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

BASE = os.environ.get("INFLUENZA_DATA_DIR", r"E:\上班后的资料\甲乙流多中心3")
OUT  = os.path.join(BASE, r'代码存档\运行结果\13_配对bootstrap')
os.makedirs(OUT, exist_ok=True)

ALL_NUM = ['age', 'WBC', 'NEU', 'NEU%', 'LYM', 'LYM%', 'BASO', 'BASO%',
           'EO', 'EO%', 'MON', 'MON%', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH',
           'MCHC', 'PLT', 'MPV', 'PCT', 'PDW', 'RDW-CV', 'month']

def pre_fit(X_tr):
    num_cols = [c for c in ALL_NUM if c != 'sex']
    pre = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), ['sex']),
    ], remainder='drop')
    pre.fit(X_tr)
    return pre

def fit_models(X_tr, y_tr, X_va):
    num_cols = [c for c in ALL_NUM if c != 'sex']
    pre = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), ['sex']),
    ], remainder='drop')
    Xt = pre.fit_transform(X_tr); Xv = pre.transform(X_va)
    # LightGBM
    m1 = lgb.LGBMClassifier(objective='multiclass', num_class=3,
                             learning_rate=0.05, n_estimators=100, num_leaves=31,
                             random_state=42, verbosity=-1)
    m1.fit(Xt, y_tr)
    p1 = m1.predict_proba(Xv)
    # XGBoost
    m2 = xgb.XGBClassifier(objective='multi:softprob', learning_rate=0.05,
                            n_estimators=100, max_depth=3,
                            random_state=42, use_label_encoder=False,
                            eval_metric='mlogloss', verbosity=0)
    m2.fit(Xt, y_tr)
    p2 = m2.predict_proba(Xv)
    # RF
    m3 = RandomForestClassifier(n_estimators=200, max_depth=20,
                                 min_samples_leaf=4, random_state=42, n_jobs=-1)
    m3.fit(Xt, y_tr)
    p3 = m3.predict_proba(Xv)
    return p1, p2, p3

def macro_auc(y, p):
    return roc_auc_score(y, p, multi_class='ovr', average='macro')

TAH = pd.read_csv(os.path.join(BASE, 'a附三院数据_processed.csv'))
X = TAH[ALL_NUM + ['sex']]; y = TAH['RESULT'].values

# 与原稿相同的70/30拆分（random_state=42）
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.30,
                                       stratify=y, random_state=42)

print('Fitting three models on 70% split, evaluating on 30% validation...')
p1, p2, p3 = fit_models(Xtr, ytr, Xva)
auc_lgb = macro_auc(yva, p1)
auc_xgb = macro_auc(yva, p2)
auc_rf  = macro_auc(yva, p3)
print(f'Point estimates (validation set, n={len(yva)}):')
print(f'  LightGBM = {auc_lgb:.4f}')
print(f'  XGBoost  = {auc_xgb:.4f}')
print(f'  RF       = {auc_rf:.4f}')

# 配对bootstrap：每次从验证集有放回抽样
rng = np.random.RandomState(42)
n = len(yva)
B = 1000
diff_lgb_xgb = np.zeros(B)
diff_lgb_rf  = np.zeros(B)
for b in range(B):
    idx = rng.randint(0, n, n)
    yb = yva[idx]
    diff_lgb_xgb[b] = macro_auc(yb, p1[idx]) - macro_auc(yb, p2[idx])
    diff_lgb_rf[b]  = macro_auc(yb, p1[idx]) - macro_auc(yb, p3[idx])

def ci(d):
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 50)), float(np.percentile(d, 97.5))

rows = [
    {'comparison': 'LightGBM vs XGBoost',
     'mean_diff': float(diff_lgb_xgb.mean()),
     'median_diff': float(np.median(diff_lgb_xgb)),
     'ci_lo': ci(diff_lgb_xgb)[0], 'ci_med': ci(diff_lgb_xgb)[1], 'ci_hi': ci(diff_lgb_xgb)[2],
     'p_lightGBM_better': float((diff_lgb_xgb > 0).mean()),
     'point_lgbm': auc_lgb, 'point_xgb': auc_xgb},
    {'comparison': 'LightGBM vs RandomForest',
     'mean_diff': float(diff_lgb_rf.mean()),
     'median_diff': float(np.median(diff_lgb_rf)),
     'ci_lo': ci(diff_lgb_rf)[0], 'ci_med': ci(diff_lgb_rf)[1], 'ci_hi': ci(diff_lgb_rf)[2],
     'p_lightGBM_better': float((diff_lgb_rf > 0).mean()),
     'point_lgbm': auc_lgb, 'point_rf': auc_rf},
]

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, 'paired_bootstrap.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 200)
print('\n=== 配对 bootstrap 结果（1000次） ===')
print(df.round(4).to_string(index=False))
print(f'\nsaved to {OUT}')
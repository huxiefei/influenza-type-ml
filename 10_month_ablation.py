# -*- coding: utf-8 -*-
"""
项目10: 月份消融分析 — 三模型对比
M1: 仅 month
M2: CBC + age + sex（剔除 month）
M4: 全变量（与原稿一致）
指标：内部5折CV macro-AUC + 外部三中心macro-AUC + 分类别OvR AUC
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import lightgbm as lgb

BASE = os.environ.get("INFLUENZA_DATA_DIR", r"E:\上班后的资料\甲乙流多中心3")
OUT  = os.path.join(BASE, r'代码存档\运行结果\10_月份消融')
os.makedirs(OUT, exist_ok=True)

# 特征集定义
ALL_NUM = ['age', 'WBC', 'NEU', 'NEU%', 'LYM', 'LYM%', 'BASO', 'BASO%',
           'EO', 'EO%', 'MON', 'MON%', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH',
           'MCHC', 'PLT', 'MPV', 'PCT', 'PDW', 'RDW-CV']
CAT_COLS = ['sex']
MONTH_COL = 'month'

def feat_sets():
    return {
        'M1_month_only': [MONTH_COL],
        'M2_CBC_age_sex': ALL_NUM + CAT_COLS,
        'M3_full': ALL_NUM + CAT_COLS + [MONTH_COL],
    }

# 公共预处理（sex 用 one-hot，数值用 standard；month 数值 z-score）
def make_preprocessor(num_cols):
    num_only = [c for c in num_cols if c != 'sex']
    has_sex = 'sex' in num_cols
    transformers = [('num', StandardScaler(), num_only)]
    if has_sex:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore',
                                                   drop='if_binary'), ['sex']))
    return ColumnTransformer(transformers, remainder='drop')

def load(path):
    df = pd.read_csv(path)
    # 调整列序
    if 'sex' in df.columns and df.columns[0] != 'sex':
        pass
    return df

def get_xy(df, num_cols):
    X = df[num_cols].copy()
    y = df['RESULT'].values
    return X, y

def lgb_params():
    return dict(
        objective='multiclass', num_class=3, metric='multi_logloss',
        learning_rate=0.05, n_estimators=100, num_leaves=31,
        random_state=42, verbosity=-1,
    )

def cv_macro_auc(X, y, num_cols, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for tr, va in skf.split(X, y):
        pre = make_preprocessor(num_cols)
        Xt = pre.fit_transform(X.iloc[tr])
        Xv = pre.transform(X.iloc[va])
        model = lgb.LGBMClassifier(**lgb_params())
        model.fit(Xt, y[tr])
        p = model.predict_proba(Xv)
        aucs.append(roc_auc_score(y[va], p, multi_class='ovr', average='macro'))
    return float(np.mean(aucs)), float(np.std(aucs))

def external_metrics(model, pre, ext_df, num_cols):
    Xe = ext_df[num_cols].copy()
    ye = ext_df['RESULT'].values
    Xt = pre.transform(Xe)
    p = model.predict_proba(Xt)
    macro = roc_auc_score(ye, p, multi_class='ovr', average='macro')
    per_class = {1: roc_auc_score((ye==1).astype(int), p[:,1]),
                 2: roc_auc_score((ye==2).astype(int), p[:,2])}
    return {'macro_auc': macro, 'FluA_auc': per_class[1], 'FluB_auc': per_class[2]}

# ============== 加载数据 ==============
TAH   = load(os.path.join(BASE, 'a附三院数据_processed.csv'))
SAH   = load(os.path.join(BASE, 'b新桥数据_processed.csv'))
CEMC  = load(os.path.join(BASE, 'c急救数据.csv'))
FDH   = load(os.path.join(BASE, 'd丰都数据_processed.csv'))

results = []
# 70/30 split for final model fit (only used for external val step)
df_tr, df_va = train_test_split(TAH, test_size=0.30, stratify=TAH['RESULT'],
                                random_state=42)

for tag, cols in feat_sets().items():
    print(f'\n>>> {tag} | {len(cols)} features')
    # 内部5折CV
    Xfull = TAH[cols].copy(); yfull = TAH['RESULT'].values
    mu, sd = cv_macro_auc(Xfull, yfull, cols)
    print(f'  Internal 5-fold macro-AUC: {mu:.4f} ± {sd:.4f}')

    # 训练最终模型（70%）并评估外部三中心
    pre = make_preprocessor(cols)
    Xt = pre.fit_transform(df_tr[cols])
    Xv = pre.transform(df_va[cols])
    Xs = pre.transform(SAH[cols])
    Xc = pre.transform(CEMC[cols])
    Xf = pre.transform(FDH[cols])
    model = lgb.LGBMClassifier(**lgb_params())
    model.fit(Xt, df_tr['RESULT'].values)
    # 内部验证集
    pv = model.predict_proba(Xv)
    in_macro = roc_auc_score(df_va['RESULT'].values, pv, multi_class='ovr', average='macro')
    print(f'  Internal validation macro-AUC: {in_macro:.4f}')
    # 外部三中心
    for name, ext, Xe in [('SAH', SAH, Xs), ('CEMC', CEMC, Xc), ('FDH', FDH, Xf)]:
        ye = ext['RESULT'].values
        p  = model.predict_proba(Xe)
        macro = roc_auc_score(ye, p, multi_class='ovr', average='macro')
        aA = roc_auc_score((ye==1).astype(int), p[:,1])
        aB = roc_auc_score((ye==2).astype(int), p[:,2])
        results.append({'model': tag, 'cohort': name, 'macro_auc': macro,
                        'FluA_auc': aA, 'FluB_auc': aB})
        print(f'  External {name}: macro={macro:.4f}, FluA={aA:.4f}, FluB={aB:.4f}')
    results.append({'model': tag, 'cohort': 'Internal_valid', 'macro_auc': in_macro,
                    'FluA_auc': np.nan, 'FluB_auc': np.nan})
    results.append({'model': tag, 'cohort': 'CV5fold', 'macro_auc': mu, 'FluA_auc': np.nan,
                    'FluB_auc': np.nan, 'std': sd})

df_res = pd.DataFrame(results)
df_res.to_csv(os.path.join(OUT, 'ablation_comparison.csv'), index=False, encoding='utf-8-sig')
print('\n=== 汇总 ===')
piv = df_res.pivot_table(index='model', columns='cohort', values='macro_auc')
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30)
print(piv.round(4).to_string())
print(f'\nsaved to {OUT}')
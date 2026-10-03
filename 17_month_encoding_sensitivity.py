# -*- coding: utf-8 -*-
"""Sensitivity check requested by R4-5: does a cyclical (sine/cosine) or categorical encoding of month of
presentation change external performance? Same hyperparameters and split as the reported model."""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

BASE = r"E:\上班后的资料\甲乙流多中心3"
NUM = ['age','WBC','NEU','NEU%','LYM','LYM%','BASO','BASO%','EO','EO%','MON','MON%','RBC','HGB','HCT',
       'MCV','MCH','MCHC','PLT','MPV','PCT','PDW','RDW-CV']
CAT = ['sex']

def load(f):
    return pd.read_csv(os.path.join(BASE, f))

TAH, SAH, CEMC, FDH = (load('a附三院数据_processed.csv'), load('b新桥数据_processed.csv'),
                       load('c急救数据.csv'), load('d丰都数据_processed.csv'))

def build(df, mode):
    d = df.copy()
    if mode == 'ordinal':
        d['month_enc'] = d['month']
        extra = ['month_enc']
    elif mode == 'cyclical':
        d['month_sin'] = np.sin(2*np.pi*(d['month']-1)/12)
        d['month_cos'] = np.cos(2*np.pi*(d['month']-1)/12)
        extra = ['month_sin','month_cos']
    elif mode == 'categorical':
        d = pd.concat([d, pd.get_dummies(d['month'].astype(int).astype('category'), prefix='m').astype(int)], axis=1)
        extra = [c for c in d.columns if c.startswith('m_')]
    return d, NUM + extra + CAT

def run(mode):
    X_all = {}
    for name, df in [('TAH',TAH),('SAH',SAH),('CEMC',CEMC),('FDH',FDH)]:
        d, cols = build(df, mode)
        X_all[name] = (d[cols], d['RESULT'].values, cols)
    cols = X_all['TAH'][2]
    Xtr, Xva, ytr, yva = train_test_split(X_all['TAH'][0], X_all['TAH'][1], test_size=0.30,
                                          stratify=X_all['TAH'][1], random_state=42)
    num_cols = [c for c in cols if c != 'sex']
    pre = ColumnTransformer([('num', StandardScaler(), num_cols),
                             ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), ['sex'])])
    pipe = Pipeline([('pre', pre), ('clf', lgb.LGBMClassifier(objective='multiclass', num_class=3,
                     learning_rate=0.05, n_estimators=100, num_leaves=31, random_state=42, verbosity=-1))])
    pipe.fit(Xtr, ytr)
    out = {'internal': roc_auc_score(yva, pipe.predict_proba(Xva), multi_class='ovr', average='macro')}
    for name in ['SAH','CEMC','FDH']:
        Xe, ye, _ = X_all[name]
        out[name] = roc_auc_score(ye, pipe.predict_proba(Xe), multi_class='ovr', average='macro')
    return out

res = {}
for mode in ['ordinal','cyclical','categorical']:
    res[mode] = run(mode)
    print(mode, {k: round(v,4) for k,v in res[mode].items()})

print('\ndifferences vs ordinal:')
for mode in ['cyclical','categorical']:
    print(mode, {k: round(res[mode][k]-res['ordinal'][k],4) for k in res['ordinal']})

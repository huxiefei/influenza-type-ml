# -*- coding: utf-8 -*-
"""逐特征计算：生理合理范围外的值（拟剔除）与推导队列 mu±3SD 外的值（缩尾）"""
import pandas as pd
import numpy as np
import json

FILES = {
    'TAH':  r'E:\上班后的资料\甲乙流多中心3\a附三院数据_processed.csv',
    'SAH':  r'E:\上班后的资料\甲乙流多中心3\b新桥数据_processed.csv',
    'CEMC': r'E:\上班后的资料\甲乙流多中心3\c急救数据.csv',
    'FDH':  r'E:\上班后的资料\甲乙流多中心3\d丰都数据_processed.csv',
}
FEATS = ['age','WBC','NEU','NEU%','LYM','LYM%','BASO','BASO%','EO','EO%',
         'MON','MON%','RBC','HGB','HCT','MCV','MCH','MCHC','PLT','MPV','PCT','PDW','RDW-CV']
UNITS = {
 'age':'years','WBC':'x10^9/L','NEU':'x10^9/L','NEU%':'%','LYM':'x10^9/L','LYM%':'%',
 'BASO':'x10^9/L','BASO%':'%','EO':'x10^9/L','EO%':'%','MON':'x10^9/L','MON%':'%',
 'RBC':'x10^12/L','HGB':'g/L','HCT':'%','MCV':'fL','MCH':'pg','MCHC':'g/L','PLT':'x10^9/L',
 'MPV':'fL','PCT':'%','PDW':'%','RDW-CV':'%',
}
# 生理合理范围（下限, 上限）——基于常规血液学参考与实验室可报告范围
PLAUS = {
 'age':(0,120),'WBC':(0.1,100),'NEU':(0,80),'NEU%':(0,100),'LYM':(0,60),'LYM%':(0,100),
 'BASO':(0,10),'BASO%':(0,100),'EO':(0,20),'EO%':(0,100),'MON':(0,20),'MON%':(0,100),
 'RBC':(0.5,9.0),'HGB':(20,250),'HCT':(5,80),'MCV':(30,160),'MCH':(5,60),'MCHC':(150,450),
 'PLT':(1,1500),'MPV':(1,30),'PCT':(0.001,2.0),'PDW':(1,50),'RDW-CV':(5,40),
}

data = {k: pd.read_csv(v) for k, v in FILES.items()}
tah = data['TAH']

rows = []
for f in FEATS:
    mu = tah[f].mean(); sd = tah[f].std()
    lo3, hi3 = mu - 3*sd, mu + 3*sd
    plo, phi = PLAUS[f]
    rec = {
        'feature': f, 'unit': UNITS[f],
        'plaus_lo': plo, 'plaus_hi': phi,
        'mu': round(float(mu), 2), 'sd': round(float(sd), 2),
        'lo3sd': round(float(lo3), 2), 'hi3sd': round(float(hi3), 2),
        'n_implaus': {}, 'pct_implaus': {}, 'n_out3sd': {}, 'pct_out3sd': {},
    }
    for c, df in data.items():
        col = df[f]
        n = len(col)
        mask_p = (col < plo) | (col > phi)
        mask_s = (col < lo3) | (col > hi3)
        rec['n_implaus'][c] = int(mask_p.sum()); rec['pct_implaus'][c] = round(100*mask_p.mean(), 3)
        rec['n_out3sd'][c] = int(mask_s.sum());  rec['pct_out3sd'][c] = round(100*mask_s.mean(), 3)
    rows.append(rec)

with open('outlier_stats.json', 'w', encoding='utf-8') as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)

# 打印摘要
print(f'{"feat":8s} {"plaus":>14s} | TAH implaus / 3SD | SAH | CEMC | FDH')
tot_p = {c:0 for c in data}; tot_s = {c:0 for c in data}
for r in rows:
    print(f"{r['feature']:8s} {str((r['plaus_lo'],r['plaus_hi'])):>14s} | "
          f"{r['n_implaus']['TAH']:5d}/{r['n_out3sd']['TAH']:5d} | "
          f"{r['n_implaus']['SAH']:5d}/{r['n_out3sd']['SAH']:5d} | "
          f"{r['n_implaus']['CEMC']:5d}/{r['n_out3sd']['CEMC']:5d} | "
          f"{r['n_implaus']['FDH']:5d}/{r['n_out3sd']['FDH']:5d}")
    for c in data:
        tot_p[c] += r['n_implaus'][c]; tot_s[c] += r['n_out3sd'][c]
print()
print('合计 生理范围外:', tot_p)
print('合计 ±3SD 外  :', tot_s)
for c, df in data.items():
    print(f'{c}: n={len(df)}')

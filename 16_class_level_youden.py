# -*- coding: utf-8 -*-
"""从 predictions.csv 计算各类别在 Youden 阈值下的敏感度/特异度（仅 numpy）"""
import json
import numpy as np
import pandas as pd

def roc(y, s):
    order = np.argsort(-s)
    y = np.asarray(y, float)[order]; s = np.asarray(s, float)[order]
    P = y.sum(); N = len(y) - P
    tpr = np.cumsum(y) / P
    fpr = np.cumsum(1 - y) / N
    return np.r_[0, fpr], np.r_[0, tpr], np.r_[s, s[-1]]

FILES = {
 'SAH':  r'E:\上班后的资料\甲乙流多中心3\附三8\外部验证\SAH_predictions.csv',
 'CEMC': r'E:\上班后的资料\甲乙流多中心3\附三8\外部验证\CEMC_predictions.csv',
 'FDH':  r'E:\上班后的资料\甲乙流多中心3\附三8\外部验证\FDH_predictions.csv',
}
CLASSES = {'FluA': ('Prob_FluA', 1), 'FluB': ('Prob_FluB', 2)}
out = {}
for center, path in FILES.items():
    df = pd.read_csv(path)
    y = df['True_Label'].values
    out[center] = {}
    for cname, (col, code) in CLASSES.items():
        p = df[col].values
        ybin = (y == code).astype(int)
        fpr, tpr, thr = roc(ybin, p)
        k = int(np.argmax(tpr - fpr))
        t_opt = float(thr[k])
        pred = (p >= t_opt).astype(int)
        tp = int(((pred == 1) & (ybin == 1)).sum()); fn = int(((pred == 0) & (ybin == 1)).sum())
        tn = int(((pred == 0) & (ybin == 0)).sum()); fp = int(((pred == 1) & (ybin == 0)).sum())
        sens = tp/(tp+fn); spec = tn/(tn+fp); ppv = tp/(tp+fp) if tp+fp else float('nan')
        npv = tn/(tn+fn) if tn+fn else float('nan')
        out[center][cname] = dict(threshold=round(t_opt,3), sens=round(sens,3), spec=round(spec,3),
                                  ppv=round(ppv,3), npv=round(npv,3))
        print(f'{center:5s} {cname}: thr={t_opt:.3f} sens={sens:.3f} spec={spec:.3f} ppv={ppv:.3f} npv={npv:.3f}')
json.dump(out, open('youden_classwise.json','w'), indent=1)

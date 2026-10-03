# -*- coding: utf-8 -*-
r"""
09_二分类指标_任意流感vs非流感.py
============================================================
回应审稿人 R5 建议：报告"任意流感 vs 非流感"二分类任务性能。
基于附三8\外部验证\*_predictions.csv（含真实标签+三类预测概率）计算：
  P(influenza) = P(FluA) + P(FluB)
  标签 = 1 if FluA or FluB else 0
指标：AUC、Brier、准确率、Sens/Spec/PPV/NPV@0.5、Youden最优阈值及对应性能
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, brier_score_loss, accuracy_score,
                             confusion_matrix)

BASE = os.environ.get("INFLUENZA_DATA_DIR", r"E:\上班后的资料\甲乙流多中心3")
OUT = os.path.join(BASE, r'代码存档\运行结果\09_二分类指标')
os.makedirs(OUT, exist_ok=True)

PRED_FILES = {
    'SAH':  os.path.join(BASE, r'附三8\外部验证\SAH_predictions.csv'),
    'CEMC': os.path.join(BASE, r'附三8\外部验证\CEMC_predictions.csv'),
    'FDH':  os.path.join(BASE, r'附三8\外部验证\FDH_predictions.csv'),
}

# 自动识别列名（不同版本列名略有差异）
def find_col(df, keys):
    for c in df.columns:
        for k in keys:
            if k.lower() in c.lower():
                return c
    raise KeyError(keys)

rows = []
for center, path in PRED_FILES.items():
    df = pd.read_csv(path)
    ycol = find_col(df, ['true', 'actual', 'label', 'y_'])
    pA = find_col(df, ['flua', 'flu_a', 'class1', 'p1'])
    pB = find_col(df, ['flub', 'flu_b', 'class2', 'p2'])
    y = df[ycol].values
    p_flu = df[pA].values + df[pB].values          # P(任意流感)
    y_bin = (y != 0).astype(int)                    # 0=非流感

    auc = roc_auc_score(y_bin, p_flu)
    brier = brier_score_loss(y_bin, p_flu)
    pred = (p_flu >= 0.5).astype(int)
    acc = accuracy_score(y_bin, pred)
    tn, fp, fn, tp = confusion_matrix(y_bin, pred).ravel()
    sens = tp / (tp + fn); spec = tn / (tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    prev = y_bin.mean()

    # Youden 最优阈值
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_bin, p_flu)
    j = tpr - fpr
    best = np.argmax(j)
    yt = thr[best]
    pred_y = (p_flu >= yt).astype(int)
    acc_y = accuracy_score(y_bin, pred_y)
    tn2, fp2, fn2, tp2 = confusion_matrix(y_bin, pred_y).ravel()

    rows.append({
        'Center': center, 'n': len(df),
        'Prevalence_anyFlu': prev,
        'AUC_binary': auc,
        'Brier_binary': brier,
        'Accuracy@0.5': acc,
        'Sensitivity@0.5': sens, 'Specificity@0.5': spec,
        'PPV@0.5': ppv, 'NPV@0.5': npv,
        'Youden_threshold': yt,
        'Sens@Youden': tp2/(tp2+fn2), 'Spec@Youden': tn2/(tn2+fp2),
        'Acc@Youden': acc_y,
        'Majority_baseline_acc': max(prev, 1-prev),
    })

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, '二分类_任意流感vs非流感.csv'), index=False,
           encoding='utf-8-sig')
pd.set_option('display.width', 200)
print(res.round(4).to_string(index=False))
print('\nsaved to', OUT)

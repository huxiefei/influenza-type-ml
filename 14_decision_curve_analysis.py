# -*- coding: utf-8 -*-
"""
项目14: 决策曲线分析 (DCA)
基于附三8 三中心 predictions.csv，对“任意流感 vs 非流感”二分类任务计算 DCA。
净收益 = TP_rate - FP_rate * (prevalence/(1-prevalence)) * weight
权重 w 表示不同阈值决策的概率；默认假设获益（流感检出）与代价（误诊）相等。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = os.environ.get("INFLUENZA_DATA_DIR", r"E:\上班后的资料\甲乙流多中心3")
OUT  = os.path.join(BASE, r'代码存档\运行结果\14_DCA')
os.makedirs(OUT, exist_ok=True)

PRED_FILES = {
    'SAH':  os.path.join(BASE, r'附三8\外部验证\SAH_predictions.csv'),
    'CEMC': os.path.join(BASE, r'附三8\外部验证\CEMC_predictions.csv'),
    'FDH':  os.path.join(BASE, r'附三8\外部验证\FDH_predictions.csv'),
}

def dca_net_benefit(y, p, thresholds):
    N = len(y)
    out = []
    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = int(((pred==1)&(y==1)).sum())
        fp = int(((pred==1)&(y==0)).sum())
        # 标准化净收益公式（Vickers 2006）
        nb = tp/N - fp/N * (t/(1-t))
        out.append(nb)
    return np.array(out)

thresholds = np.linspace(0.01, 0.99, 99)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
all_nb = []
for ax, (center, fp) in zip(axes, PRED_FILES.items()):
    df = pd.read_csv(fp)
    y = (df['True_Label'] != 0).astype(int).values
    p = (df['Prob_FluA'] + df['Prob_FluB']).values
    prev = y.mean()
    nb_model = dca_net_benefit(y, p, thresholds)
    # Treat-all baseline: net benefit = prev - (1-prev) * (t/(1-t))
    nb_all = prev - (1-prev) * (thresholds/(1-thresholds))
    nb_none = np.zeros_like(thresholds)
    ax.plot(thresholds, nb_model, color='#C00000', lw=2.2, label='LightGBM (any-flu)')
    ax.plot(thresholds, nb_all, color='gray', lw=1.5, ls='--', label='Treat-all')
    ax.plot(thresholds, nb_none, color='black', lw=1.2, label='Treat-none')
    ax.set_xlim(0, 0.6); ax.set_ylim(-0.05, max(0.5, nb_model.max()*1.1))
    ax.set_xlabel('Threshold probability')
    ax.set_ylabel('Net benefit')
    ax.set_title(f'{center} (prev={prev:.2%}, n={len(y)})')
    ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=0.3)
    # 标记关键阈值（Youden）
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    yt = thr[np.argmax(tpr-fpr)]
    ax.axvline(yt, color='blue', lw=1, ls=':', alpha=0.6,
               label=f'Youden={yt:.3f}')
    all_nb.append({'Center': center, 'Youden_threshold': float(yt),
                   'NB_at_Youden': float(nb_model[np.argmin(np.abs(thresholds-yt))]),
                   'NB_at_0.5':    float(nb_model[np.argmin(np.abs(thresholds-0.5))]),
                   'NB_max':       float(nb_model.max()),
                   'NB_at_threshold_max': float(thresholds[nb_model.argmax()])})

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'DCA_three_centers.png'), dpi=300, bbox_inches='tight')
plt.close()

df_nb = pd.DataFrame(all_nb)
df_nb.to_csv(os.path.join(OUT, 'DCA_at_key_thresholds.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 200)
print('=== 三中心 DCA 净收益关键点 ===')
print(df_nb.round(4).to_string(index=False))
print(f'\nsaved to {OUT}')
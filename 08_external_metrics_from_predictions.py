# -*- coding: utf-8 -*-
"""
08_补充外部验证指标_基于predictions.py
============================================================
功能：直接基于外部验证 predictions.csv（不重训模型）补算审稿人要求的
      全部补充指标：
        - 多分类 & 各类 Brier 分数（R4-7 / R1-M6）
        - 各中心三分类混淆矩阵（R5-M4）
        - FluA / FluB 二分类（OvR）灵敏度/特异度/PPV/NPV/Youden（R5-M2）
        - 基线对比：多数类基线与先验概率基线（R5-M4）
        - 类别患病率（基线特征补充）
数据来源优先级：代码存档运行结果 → 附三8\外部验证（原始产物）

输出：代码存档/运行结果/08_补充外部指标/
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (brier_score_loss, confusion_matrix,
                             precision_recall_curve, roc_curve, auc)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置区 ====================
BASE_DIR = os.environ.get("INFLUENZA_DATA_DIR", r"E:\上班后的资料\甲乙流多中心3")
PRED_CANDIDATES = [
    os.path.join(BASE_DIR, r'代码存档\运行结果\04_外部验证'),
    os.path.join(BASE_DIR, r'附三8\外部验证'),
]
OUTPUT_DIR = os.path.join(BASE_DIR, r'代码存档\运行结果\08_补充外部指标')
CENTERS = ['SAH', 'CEMC', 'FDH']
CLASS_NAME = {0: 'non-Flu', 1: 'FluA', 2: 'FluB'}
PROB_COLS = {0: 'Prob_nonFlu', 1: 'Prob_FluA', 2: 'Prob_FluB'}
# ================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

records, cm_records = [], []


def find_predictions(center):
    for d in PRED_CANDIDATES:
        p = os.path.join(d, f'{center}_predictions.csv')
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'{center}_predictions.csv 未找到')


for center in CENTERS:
    path = find_predictions(center)
    df = pd.read_csv(path, encoding='utf-8-sig')
    y = df['True_Label'].values
    pred = df['Predicted_Label'].values
    proba = df[['Prob_nonFlu', 'Prob_FluA', 'Prob_FluB']].values
    n = len(y)
    print(f"\n===== {center} (n={n:,}) =====")

    # --- 1. 患病率与基线 ---
    prev = {CLASS_NAME[c]: (y == c).mean() for c in range(3)}
    majority_acc = max((y == c).mean() for c in range(3))   # 多数类基线准确率
    prior_brier = np.mean([1 - prev[CLASS_NAME[c]] for c in y])  # 先验概率基线Brier

    # --- 2. 多分类 Brier ---
    brier_multi = np.mean(np.sum((proba - np.eye(3)[y]) ** 2, axis=1))
    acc = (y == pred).mean()
    print(f"  准确率 {acc:.4f} | 多数类基线 {majority_acc:.4f} "
          f"(提升 {acc - majority_acc:+.4f})")
    print(f"  多分类Brier {brier_multi:.4f} | 先验基线Brier {prior_brier:.4f}")

    records.append({'Center': center, 'n': n, 'Metric': 'Accuracy', 'Value': acc})
    records.append({'Center': center, 'n': n, 'Metric': 'Majority_baseline_acc',
                    'Value': majority_acc})
    records.append({'Center': center, 'n': n, 'Metric': 'Multiclass_Brier',
                    'Value': brier_multi})
    records.append({'Center': center, 'n': n, 'Metric': 'Prior_baseline_Brier',
                    'Value': prior_brier})
    for c in range(3):
        records.append({'Center': center, 'n': n,
                        'Metric': f'Prevalence_{CLASS_NAME[c]}', 'Value': prev[CLASS_NAME[c]]})

    # --- 3. 各类 Brier + 二分类指标 ---
    for c in [1, 2]:   # FluA / FluB
        cname = CLASS_NAME[c]
        y_bin = (y == c).astype(int)
        p_bin = proba[:, c]

        b = brier_score_loss(y_bin, p_bin)
        fpr, tpr, thr = roc_curve(y_bin, p_bin)
        roc_auc_v = auc(fpr, tpr)
        j_idx = np.argmax(tpr - fpr)
        youden_thr = thr[j_idx]
        yhat = (p_bin >= 0.5).astype(int)          # 默认0.5阈值
        tp = int(((yhat == 1) & (y_bin == 1)).sum())
        fp = int(((yhat == 1) & (y_bin == 0)).sum())
        fn = int(((yhat == 0) & (y_bin == 1)).sum())
        tn = int(((yhat == 0) & (y_bin == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        ppv = tp / (tp + fp) if (tp + fp) else np.nan
        npv = tn / (tn + fn) if (tn + fn) else np.nan

        print(f"  {cname}: Brier={b:.4f}, OvR-AUC={roc_auc_v:.4f}, "
              f"Sens={sens:.3f}, Spec={spec:.3f}, PPV={ppv:.3f}, NPV={npv:.3f}")

        for k, v in [(f'Brier_{cname}', b),
                     (f'OvR_AUC_{cname}', roc_auc_v),
                     (f'Sensitivity_{cname}@0.5', sens),
                     (f'Specificity_{cname}@0.5', spec),
                     (f'PPV_{cname}@0.5', ppv),
                     (f'NPV_{cname}@0.5', npv),
                     (f'Youden_threshold_{cname}', youden_thr)]:
            records.append({'Center': center, 'n': n, 'Metric': k, 'Value': v})

    # --- 4. 三分类混淆矩阵 ---
    cm = confusion_matrix(y, pred)
    cm_records.append((center, cm))
    pd.DataFrame(cm, index=[f'True_{CLASS_NAME[i]}' for i in range(3)],
                 columns=[f'Pred_{CLASS_NAME[i]}' for i in range(3)]
                 ).to_csv(os.path.join(OUTPUT_DIR, f'{center}_confusion_matrix.csv'),
                          encoding='utf-8-sig')

# --- 汇总长表 → 透视表 ---
res_df = pd.DataFrame(records)
res_df.to_csv(os.path.join(OUTPUT_DIR, '外部验证补充指标_长表.csv'),
              index=False, encoding='utf-8-sig')
pivot = res_df.pivot_table(index='Metric', columns='Center', values='Value')
pivot.to_csv(os.path.join(OUTPUT_DIR, '外部验证补充指标_汇总表.csv'),
             encoding='utf-8-sig')
print("\n" + pivot.round(4).to_string())

# --- 混淆矩阵三联图 ---
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, (center, cm) in zip(axes, cm_records):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['non-Flu', 'FluA', 'FluB'],
                yticklabels=['non-Flu', 'FluA', 'FluB'])
    ax.set_title(f'{center} (n={cm.sum():,})'); ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '三中心混淆矩阵.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n📁 结果已保存至: {OUTPUT_DIR}")

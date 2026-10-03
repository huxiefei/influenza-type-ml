# -*- coding: utf-8 -*-
"""
05_校准曲线_PR曲线_Brier.py
============================================================
功能：三中心外部验证的校准曲线（n_bins=10 + 样条平滑）、PR 曲线、
      Brier 分数 → 对应文章 Figure 2 / eTable 7 校准部分
      【修复】原附三7代码 Brier 算了没保存 —— 本脚本全部落盘

数据来源：04 脚本生成的 *_predictions.csv（不重训模型）

输出：代码存档/运行结果/05_校准与PR/
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, average_precision_score, brier_score_loss
from scipy.interpolate import UnivariateSpline

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置区 ====================
BASE_DIR = os.environ.get("INFLUENZA_DATA_DIR", r"E:\上班后的资料\甲乙流多中心3")
PRED_DIR = os.path.join(BASE_DIR, r'代码存档\运行结果\04_外部验证')
OUTPUT_DIR = os.path.join(BASE_DIR, r'代码存档\运行结果\05_校准与PR')

CENTERS = [('SAH', 'SAH_predictions.csv'),
           ('CEMC', 'CEMC_predictions.csv'),
           ('FDH', 'FDH_predictions.csv')]
CLASS_PROB_COL = {1: 'Prob_FluA', 2: 'Prob_FluB'}
CLASS_NAME = {0: 'non-Flu', 1: 'FluA', 2: 'FluB'}
N_BINS = 10
SMOOTH_S = 0.8   # 样条平滑因子
# ================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
center_colors = {'SAH': '#1f77b4', 'CEMC': '#ff7f0e', 'FDH': '#2ca02c'}

# 读取三中心预测明细
data = {}
for name_en, fname in CENTERS:
    path = os.path.join(PRED_DIR, fname)
    if not os.path.exists(path):
        # 回退：使用原始附三8的预测文件
        path = os.path.join(BASE_DIR, r'附三8\外部验证', fname)
    df = pd.read_csv(path, encoding='utf-8-sig')
    data[name_en] = df
    print(f"✅ {name_en}: {len(df):,} 样本 ({path})")

records = []

# ============================================================
# 1. 校准曲线（FluA / FluB，n_bins=10 + 样条平滑）
# ============================================================
for cls in [1, 2]:
    plt.figure(figsize=(8, 7))
    for name_en, df in data.items():
        y_bin = (df['True_Label'] == cls).astype(int)
        prob = df[CLASS_PROB_COL[cls]]
        frac_pos, mean_pred = calibration_curve(y_bin, prob, n_bins=N_BINS,
                                                strategy='uniform')
        # 样条平滑
        order = np.argsort(mean_pred)
        if len(mean_pred) >= 4:
            try:
                spline = UnivariateSpline(mean_pred[order], frac_pos[order], s=SMOOTH_S, k=3)
                xs = np.linspace(mean_pred.min(), min(mean_pred.max(), 1.0), 200)
                ys = spline(xs)
                ys = np.clip(ys, 0, 1)
            except Exception:
                xs, ys = mean_pred[order], frac_pos[order]
        else:
            xs, ys = mean_pred[order], frac_pos[order]

        plt.plot(xs, ys, lw=2, color=center_colors[name_en],
                 label=f'{name_en} (smoothed)')
        plt.scatter(mean_pred, frac_pos, color=center_colors[name_en], s=30, zorder=5)

    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
    plt.xlabel('Mean predicted probability', fontsize=12)
    plt.ylabel('Fraction of positives (observed)', fontsize=12)
    plt.title(f'Calibration Curves — {CLASS_NAME[cls]} (External Validation, {N_BINS} bins + spline)',
              fontweight='bold')
    plt.legend(loc='upper left'); plt.grid(True, alpha=0.3)
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'Combined_Calibration_Curve_{CLASS_NAME[cls]}_Smooth.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 校准曲线已保存: {CLASS_NAME[cls]}")

# ============================================================
# 2. PR 曲线（FluA / FluB）
# ============================================================
for cls in [1, 2]:
    plt.figure(figsize=(8, 7))
    for name_en, df in data.items():
        y_bin = (df['True_Label'] == cls).astype(int)
        prob = df[CLASS_PROB_COL[cls]]
        prec, rec, _ = precision_recall_curve(y_bin, prob)
        ap = average_precision_score(y_bin, prob)
        plt.plot(rec, prec, lw=2, color=center_colors[name_en],
                 label=f'{name_en} (AP = {ap:.3f})')
        records.append({'Center': name_en, 'Class': CLASS_NAME[cls],
                        'PR_AUC(AP)': ap})
    plt.xlabel('Recall', fontsize=12); plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curves — {CLASS_NAME[cls]} (External Validation)',
              fontweight='bold')
    plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)
    plt.xlim([0, 1]); plt.ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'Combined_PR_Curve_{CLASS_NAME[cls]}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ PR曲线已保存: {CLASS_NAME[cls]}")

# ============================================================
# 3. Brier 分数（多分类 + 各类 OvR）—— 落盘修复
# ============================================================
for name_en, df in data.items():
    y_true = df['True_Label']
    proba = df[['Prob_nonFlu', 'Prob_FluA', 'Prob_FluB']].values
    # 多分类 Brier（类别概率之和为1时等于各类 OvR Brier 之和）
    brier_multi = np.mean(np.sum((proba - np.eye(3)[y_true]) ** 2, axis=1))
    records.append({'Center': name_en, 'Class': 'Multiclass(sum)', 'PR_AUC(AP)': np.nan,
                    'Brier': brier_multi})
    for cls, col in [(0, 'Prob_nonFlu'), (1, 'Prob_FluA'), (2, 'Prob_FluB')]:
        b = brier_score_loss((y_true == cls).astype(int), df[col])
        records.append({'Center': name_en, 'Class': CLASS_NAME[cls],
                        'Brier': b})

brier_df = pd.DataFrame(records)
brier_df.to_csv(os.path.join(OUTPUT_DIR, '校准PR_Brier汇总.csv'),
                index=False, encoding='utf-8-sig')

print("\n" + brier_df.to_string(index=False))
print(f"\n📁 结果已保存至: {OUTPUT_DIR}")

# -*- coding: utf-8 -*-
"""
03_训练最佳模型_LightGBM.py
============================================================
功能：用已选定的最优超参训练最终 LightGBM 模型（TAH 推导队列, 70:30），
      输出内部验证全套指标与模型文件 → 对应文章 Table 2 / eTable 7 内部部分
最终超参（与文章 eTable 7 一致，勿改）:
      learning_rate=0.05, n_estimators=100, num_leaves=31, random_state=42

输出：代码存档/运行结果/03_最佳模型/
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (roc_curve, auc, classification_report,
                             confusion_matrix, accuracy_score, brier_score_loss)
from lightgbm import LGBMClassifier

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置区 ====================
BASE_DIR = os.environ.get("INFLUENZA_DATA_DIR", r"E:\上班后的资料\甲乙流多中心3")
MODELING_DATA = os.path.join(BASE_DIR, 'a附三院数据_processed.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, r'代码存档\运行结果\03_最佳模型')
RANDOM_STATE = 42

BEST_PARAMS = {
    'learning_rate': 0.05,
    'n_estimators': 100,
    'num_leaves': 31,
    'random_state': RANDOM_STATE,
    'verbose': -1,
}
# ================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
class_names = {0: 'non-Flu', 1: 'FluA', 2: 'FluB'}
target_names = [class_names[i] for i in sorted(class_names)]

df = pd.read_csv(MODELING_DATA, encoding='utf-8-sig')
X, y = df.drop('RESULT', axis=1), df['RESULT']

# --- 预处理（sex 分类，month 数值标准化；仅 fit 训练集）---
all_numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
final_categorical_cols = [c for c in ['sex'] if c in X.columns]
final_numeric_cols = [c for c in all_numeric_cols if c not in final_categorical_cols]
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), final_numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), final_categorical_cols)
    ], remainder='passthrough')

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
print(f"训练集 {len(X_train):,} / 内部验证 {len(X_val):,}")

X_train_p = preprocessor.fit_transform(X_train)
X_val_p = preprocessor.transform(X_val)

# --- 训练最终模型 ---
final_model = LGBMClassifier(**BEST_PARAMS)
final_model.fit(X_train_p, y_train)

y_pred = final_model.predict(X_val_p)
y_proba = final_model.predict_proba(X_val_p)
accuracy = accuracy_score(y_val, y_pred)

fpr, tpr, roc_auc = {}, {}, {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_val, y_proba[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
mean_tpr = np.mean([np.interp(all_fpr, fpr[i], tpr[i]) for i in range(3)], axis=0)
roc_auc['macro'] = auc(all_fpr, mean_tpr)

# Brier 分数（各类 One-vs-Rest）——原代码算了没保存，这里固定保存
brier_scores = {
    class_names[c]: brier_score_loss((y_val == c).astype(int), y_proba[:, c])
    for c in range(3)}

# --- 保存模型三件套 ---
with open(os.path.join(OUTPUT_DIR, 'final_lightgbm_model.pkl'), 'wb') as f:
    pickle.dump(final_model, f)
with open(os.path.join(OUTPUT_DIR, 'preprocessor.pkl'), 'wb') as f:
    pickle.dump(preprocessor, f)
with open(os.path.join(OUTPUT_DIR, 'feature_names.pkl'), 'wb') as f:
    pickle.dump(list(X.columns), f)

# --- 图：ROC / 混淆矩阵 / 分类报告 ---
plt.figure(figsize=(10, 8))
line_colors = ['#1f77b4', '#d62728', '#2ca02c']
for i in range(3):
    plt.plot(fpr[i], tpr[i], lw=2, color=line_colors[i],
             label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
plt.plot(all_fpr, mean_tpr, 'k--', lw=2,
         label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})')
plt.plot([0, 1], [0, 1], 'gray', linestyle=':')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Final LightGBM Model: Internal Validation ROC Curves', fontweight='bold')
plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves.png'), dpi=300, bbox_inches='tight')
plt.close()

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix (Internal Validation)'); plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

report = classification_report(y_val, y_pred, target_names=target_names)

# --- 文字报告 ---
with open(os.path.join(OUTPUT_DIR, 'final_model_report.txt'), 'w', encoding='utf-8') as f:
    f.write("最终 LightGBM 模型报告（TAH 推导队列）\n" + "=" * 50 + "\n\n")
    f.write(f"模型参数: {BEST_PARAMS}\n")
    f.write(f"训练集: {len(X_train):,} 行 | 内部验证集: {len(X_val):,} 行 (70:30, 分层, seed={RANDOM_STATE})\n\n")
    f.write(f"验证集准确率: {accuracy:.4f}\n")
    for i in range(3):
        f.write(f"{class_names[i]} AUC: {roc_auc[i]:.4f}\n")
    f.write(f"宏平均 AUC: {roc_auc['macro']:.4f}\n\n")
    f.write("Brier 分数 (One-vs-Rest):\n")
    for k, v in brier_scores.items():
        f.write(f"  {k}: {v:.4f}\n")
    f.write("\n分类报告:\n" + report)

pd.DataFrame({'Brier_OvR': brier_scores}).to_csv(
    os.path.join(OUTPUT_DIR, 'brier_scores_internal.csv'), encoding='utf-8-sig')

print(f"准确率: {accuracy:.4f} | 宏平均AUC: {roc_auc['macro']:.4f}")
print(f"📁 结果已保存至: {OUTPUT_DIR}")

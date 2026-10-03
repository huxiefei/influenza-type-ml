# -*- coding: utf-8 -*-
"""
04_外部验证_三中心_含可解释性.py
============================================================
功能：用最终 LightGBM 模型对 SAH(新桥)/CEMC(急救)/FDH(丰都) 做外部验证
      + 内置重要性 & 置换重要性 → 对应文章 Table 3 / eTable 7 外部部分 /
      eFigure 分类AUC对比 + 可解释性分析
      （每样本预测明细存 *_predictions.csv，供 05/08 脚本复用）

输出：代码存档/运行结果/04_外部验证/
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, classification_report,
                             confusion_matrix, accuracy_score)
from sklearn.inspection import permutation_importance

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置区 ====================
BASE_DIR = os.environ.get("INFLUENZA_DATA_DIR", r"E:\上班后的资料\甲乙流多中心3")
MODEL_DIR = os.path.join(BASE_DIR, r'代码存档\运行结果\03_最佳模型')
OUTPUT_DIR = os.path.join(BASE_DIR, r'代码存档\运行结果\04_外部验证')

EXTERNAL_DATA = {
    '新桥': ('SAH', os.path.join(BASE_DIR, 'b新桥数据_processed.csv')),
    '急救': ('CEMC', os.path.join(BASE_DIR, 'c急救数据.csv')),
    '丰都': ('FDH', os.path.join(BASE_DIR, 'd丰都数据_processed.csv')),
}

# 用于置换重要性的数据（推导队列内部验证集重建）
INTERNAL_DATA = os.path.join(BASE_DIR, 'a附三院数据_processed.csv')
RANDOM_STATE = 42
N_REPEATS = 10
# ================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
class_names = {0: 'non-Flu', 1: 'FluA', 2: 'FluB'}
target_names = [class_names[i] for i in sorted(class_names)]

# --- 加载模型三件套 ---
with open(os.path.join(MODEL_DIR, 'final_lightgbm_model.pkl'), 'rb') as f:
    final_model = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'preprocessor.pkl'), 'rb') as f:
    preprocessor = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'feature_names.pkl'), 'rb') as f:
    training_feature_names = pickle.load(f)


def validate_external(name_cn, name_en, data_path):
    print(f"\n{'='*60}\n外部验证: {name_cn} ({name_en})\n{'='*60}")
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    y_ext = df['RESULT']
    X_ext = df.drop('RESULT', axis=1)

    # 特征对齐（缺列补0，多列丢弃，按训练顺序重排）
    for feat in training_feature_names:
        if feat not in X_ext.columns:
            X_ext[feat] = 0
    X_ext = X_ext[training_feature_names]

    X_ext_p = preprocessor.transform(X_ext)
    y_pred = final_model.predict(X_ext_p)
    y_proba = final_model.predict_proba(X_ext_p)

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_ext, y_proba[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
    mean_tpr = np.mean([np.interp(all_fpr, fpr[i], tpr[i]) for i in range(3)], axis=0)
    fpr['macro'], tpr['macro'] = all_fpr, mean_tpr
    roc_auc['macro'] = auc(all_fpr, mean_tpr)
    acc = accuracy_score(y_ext, y_pred)

    print(f"  n={len(X_ext):,} | Acc={acc:.4f} | Macro AUC={roc_auc['macro']:.4f}")
    print(f"  AUC: " + ", ".join(f"{class_names[i]}={roc_auc[i]:.4f}" for i in range(3)))

    # 每样本预测明细（供校准/PR/补充指标脚本复用）
    pred_df = pd.DataFrame({
        'True_Label': y_ext.values,
        'Predicted_Label': y_pred,
        'Prob_nonFlu': y_proba[:, 0],
        'Prob_FluA': y_proba[:, 1],
        'Prob_FluB': y_proba[:, 2]})
    pred_df.to_csv(os.path.join(OUTPUT_DIR, f'{name_en}_predictions.csv'),
                   index=False, encoding='utf-8-sig')

    # 单中心报告 + 混淆矩阵
    with open(os.path.join(OUTPUT_DIR, f'{name_en}_external_validation_report.txt'),
              'w', encoding='utf-8') as f:
        f.write(f"外部验证报告: {name_cn} ({name_en})\n" + "=" * 40 + "\n")
        f.write(f"样本量: {len(X_ext):,}\n准确率: {acc:.4f}\n")
        for i in range(3):
            f.write(f"{class_names[i]} AUC: {roc_auc[i]:.4f}\n")
        f.write(f"宏平均 AUC: {roc_auc['macro']:.4f}\n\n")
        f.write(classification_report(y_ext, y_pred, target_names=target_names))
        f.write("\n混淆矩阵:\n" + str(confusion_matrix(y_ext, y_pred)))

    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_ext, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'External Validation: Confusion Matrix ({name_en})')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{name_en}_confusion_matrix.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    return {'name_cn': name_cn, 'name_en': name_en, 'y_true': y_ext,
            'y_pred': y_pred, 'y_proba': y_proba, 'acc': acc,
            'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr}


# --- 逐中心验证 ---
validation_results = {}
for name_cn, (name_en, path) in EXTERNAL_DATA.items():
    validation_results[name_cn] = validate_external(name_cn, name_en, path)

# --- 汇总表 ---
summary_rows = []
for name_cn, res in validation_results.items():
    summary_rows.append({
        'Center': res['name_en'], 'n': len(res['y_true']),
        'Accuracy': res['acc'],
        'AUC_nonFlu': res['roc_auc'][0],
        'AUC_FluA': res['roc_auc'][1],
        'AUC_FluB': res['roc_auc'][2],
        'AUC_Macro': res['roc_auc']['macro']})
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, '三中心外部验证性能汇总表.csv'),
                  index=False, encoding='utf-8-sig')
print("\n" + summary_df.to_string(index=False))

# --- 综合图1：宏平均 ROC 对比 ---
plt.figure(figsize=(10, 8))
for res in validation_results.values():
    plt.plot(res['fpr']['macro'], res['tpr']['macro'],
             lw=2, label=f"{res['name_en']} (AUC = {res['roc_auc']['macro']:.3f})")
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('External Validation: Macro-average ROC Comparison (SAH, CEMC, FDH)',
          fontsize=14, fontweight='bold')
plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Combined_Macro_ROC_Comparison.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# --- 综合图2：各中心分类 AUC 对比（对应文章 eFigure）---
colors3 = ['#1f77b4', '#ff7f0e', '#2ca02c']
x = np.arange(len(summary_df))
bar_width = 0.25
plt.figure(figsize=(10, 6))
for i, col in enumerate(['AUC_nonFlu', 'AUC_FluA', 'AUC_FluB']):
    offset = (i - 1) * bar_width
    plt.bar(x + offset, summary_df[col], bar_width,
            label=col.replace('AUC_', ''), color=colors3[i])
    for j, v in enumerate(summary_df[col]):
        plt.text(j + offset, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
plt.xticks(x, summary_df['Center'])
plt.xlabel('External Center'); plt.ylabel('AUC')
plt.title('Class-wise AUC Comparison Across Centers', fontweight='bold')
plt.legend(title='Class'); plt.ylim(0.4, 1.05); plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'Classwise_AUC_Comparison_By_Center.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 可解释性分析：内置重要性 + 置换重要性（TAH 内部验证集）
# ============================================================
print("\n可解释性分析（内置 + 置换重要性）...")
from sklearn.model_selection import train_test_split

df_int = pd.read_csv(INTERNAL_DATA, encoding='utf-8-sig')
X_int, y_int = df_int.drop('RESULT', axis=1), df_int['RESULT']
X_tr, X_va, y_tr, y_va = train_test_split(
    X_int, y_int, test_size=0.3, stratify=y_int, random_state=RANDOM_STATE)
X_va_p = preprocessor.transform(X_va[training_feature_names])

feature_names_p = preprocessor.get_feature_names_out()

# 内置重要性（gain）
internal_imp = pd.Series(final_model.feature_importances_, index=feature_names_p)

# 置换重要性（宏平均 AUC）
perm = permutation_importance(final_model, X_va_p, y_va,
                              scoring='roc_auc_ovr',
                              n_repeats=N_REPEATS,
                              random_state=RANDOM_STATE, n_jobs=-1)

imp_df = pd.DataFrame({
    'Feature': feature_names_p,
    'Internal_importance': internal_imp.values,
    'Perm_importance_mean': perm.importances_mean,
    'Perm_importance_std': perm.importances_std})
imp_df['Internal_rank'] = imp_df['Internal_importance'].rank(ascending=False).astype(int)
imp_df['Perm_rank'] = imp_df['Perm_importance_mean'].rank(ascending=False).astype(int)
imp_df = imp_df.sort_values('Internal_importance', ascending=False)
imp_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance_internal_validation.csv'),
              index=False, encoding='utf-8-sig')

top_n = min(15, len(imp_df))
plt.figure(figsize=(10, 8))
plt.barh(imp_df['Feature'].head(top_n)[::-1],
         imp_df['Internal_importance'].head(top_n)[::-1], color='steelblue', alpha=0.85)
plt.xlabel('LightGBM Gain Importance')
plt.title('Top Feature Importance (Internal Validation, TAH)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()

with open(os.path.join(OUTPUT_DIR, 'feature_importance_report.txt'), 'w',
          encoding='utf-8') as f:
    f.write("特征重要性报告（内置 gain + 置换重要性, TAH 内部验证集, n_repeats=%d）\n" % N_REPEATS)
    f.write("=" * 60 + "\n\n")
    f.write(imp_df.to_string(index=False))

print(f"📁 结果已保存至: {OUTPUT_DIR}")

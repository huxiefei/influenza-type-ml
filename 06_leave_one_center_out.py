# -*- coding: utf-8 -*-
"""
06_各中心轮流推导_eFigure1.py
============================================================
功能：预实验——四个中心轮流作为推导队列，六模型网格搜索选最优，
      其余三中心作外部验证 → 对应文章 Methods "Derivation Cohort Selection"
      与补充材料 eFigure 1（12组 Acc/AUC）
注意：TAH 轮即正式主分析（与02/03脚本等价）；本脚本对 FDH 轮可能选出
      XGBoost（与 eFigure 1 一致，属正常现象）。

输出：代码存档/运行结果/06_各中心轮流推导/
        TAH推导/ SAH推导/ CEMC推导/ FDH推导/
        eFigure1_summary.csv（汇总12组数字）
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置区 ====================
BASE_DIR = os.environ.get("INFLUENZA_DATA_DIR", r"E:\上班后的资料\甲乙流多中心3")
OUTPUT_DIR = os.path.join(BASE_DIR, r'代码存档\运行结果\06_各中心轮流推导')
RANDOM_STATE = 42

CENTER_DATA = {
    'TAH':  ('附三院', os.path.join(BASE_DIR, 'a附三院数据_processed.csv')),
    'SAH':  ('新桥',   os.path.join(BASE_DIR, 'b新桥数据_processed.csv')),
    'CEMC': ('急救',   os.path.join(BASE_DIR, 'c急救数据.csv')),
    'FDH':  ('丰都',   os.path.join(BASE_DIR, 'd丰都数据_processed.csv')),
}

model_configs = {
    "RandomForest": (RandomForestClassifier(random_state=42),
                     {'n_estimators': [100, 200], 'max_depth': [10, 20],
                      'min_samples_leaf': [1, 4]}),
    "LightGBM": (LGBMClassifier(random_state=42, verbose=-1),
                 {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1],
                  'num_leaves': [31, 50]}),
    "XGBoost": (XGBClassifier(random_state=42, eval_metric='logloss'),
                {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1],
                 'max_depth': [3, 6]}),
    "LogisticRegression": (LogisticRegression(random_state=42, max_iter=1000),
                           {'C': [0.1, 1, 10]}),
    "AdaBoost": (AdaBoostClassifier(random_state=42),
                 {'n_estimators': [50, 100], 'learning_rate': [0.5, 1.0]}),
    "SVM": (SVC(probability=True, random_state=42),
            {'C': [1, 10], 'kernel': ['rbf', 'linear']}),
}
# ================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def make_preprocessor(X):
    all_numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in ['sex'] if c in X.columns]
    num_cols = [c for c in all_numeric if c not in cat_cols]
    return ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
    ], remainder='passthrough')


def macro_auc(y_true, y_proba):
    fpr, tpr, aucs = {}, {}, []
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_true, y_proba[:, i], pos_label=i)
        aucs.append(auc(fpr[i], tpr[i]))
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
    mean_tpr = np.mean([np.interp(all_fpr, fpr[i], tpr[i]) for i in range(3)], axis=0)
    return auc(all_fpr, mean_tpr), aucs


summary_rows = []

# ============ 逐中心作为推导队列 ============
for dev_key, (dev_cn, dev_path) in CENTER_DATA.items():
    dev_dir = os.path.join(OUTPUT_DIR, f'{dev_key}推导')
    for sub in ['模型构建', '最佳模型', '外部验证']:
        os.makedirs(os.path.join(dev_dir, sub), exist_ok=True)

    print(f"\n{'#'*60}\n# 推导队列: {dev_cn} ({dev_key})\n{'#'*60}")
    df = pd.read_csv(dev_path, encoding='utf-8-sig')
    X, y = df.drop('RESULT', axis=1), df['RESULT']
    feat_names = list(X.columns)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
    prep = make_preprocessor(X)
    X_tr_p = prep.fit_transform(X_tr)
    X_va_p = prep.transform(X_va)

    # --- 六模型网格搜索 ---
    best_name, best_model, best_auc = None, None, -1
    model_records = []
    for name, (model, grid) in model_configs.items():
        gs = GridSearchCV(model, grid, cv=cv, scoring='roc_auc_ovr', n_jobs=-1)
        gs.fit(X_tr_p, y_tr)
        y_proba = gs.best_estimator_.predict_proba(X_va_p)
        m_auc, _ = macro_auc(y_va, y_proba)
        acc = accuracy_score(y_va, gs.best_estimator_.predict(X_va_p))
        model_records.append((name, gs.best_params_, acc, m_auc))
        print(f"  {name}: Acc={acc:.4f}, MacroAUC={m_auc:.4f}")
        if m_auc > best_auc:
            best_name, best_model, best_auc = name, gs.best_estimator_, m_auc

    with open(os.path.join(dev_dir, '模型构建', 'model_selection_report.txt'),
              'w', encoding='utf-8') as f:
        f.write(f"推导队列: {dev_cn} ({dev_key}) 六模型比较\n" + "=" * 50 + "\n")
        for name, params, acc, m_auc in model_records:
            f.write(f"{name}: Acc={acc:.4f}, MacroAUC={m_auc:.4f}, params={params}\n")
        f.write(f"\n最优模型: {best_name}")

    with open(os.path.join(dev_dir, '最佳模型', 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    with open(os.path.join(dev_dir, '最佳模型', 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(prep, f)
    with open(os.path.join(dev_dir, '最佳模型', 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feat_names, f)

    y_pred = best_model.predict(X_va_p)
    y_proba = best_model.predict_proba(X_va_p)
    int_auc, int_aucs = macro_auc(y_va, y_proba)
    int_acc = accuracy_score(y_va, y_pred)
    summary_rows.append({'Derivation': dev_key, 'Validation': f'{dev_key}(内部)',
                         'BestModel': best_name, 'n': len(y_va),
                         'Accuracy': int_acc, 'MacroAUC': int_auc})
    print(f"  ★ 最优: {best_name} | 内部验证 Acc={int_acc:.4f}, MacroAUC={int_auc:.4f}")

    # --- 其余三中心外部验证 ---
    for val_key, (val_cn, val_path) in CENTER_DATA.items():
        if val_key == dev_key:
            continue
        df_v = pd.read_csv(val_path, encoding='utf-8-sig')
        y_v = df_v['RESULT']
        X_v = df_v.drop('RESULT', axis=1)
        for c in feat_names:
            if c not in X_v.columns:
                X_v[c] = 0
        X_v = X_v[feat_names]
        X_v_p = prep.transform(X_v)

        y_pred_v = best_model.predict(X_v_p)
        y_proba_v = best_model.predict_proba(X_v_p)
        v_auc, v_aucs = macro_auc(y_v, y_proba_v)
        v_acc = accuracy_score(y_v, y_pred_v)
        summary_rows.append({'Derivation': dev_key, 'Validation': val_key,
                             'BestModel': best_name, 'n': len(y_v),
                             'Accuracy': v_acc, 'MacroAUC': v_auc})
        print(f"    → 验证 {val_key}: Acc={v_acc:.4f}, MacroAUC={v_auc:.4f}")

        with open(os.path.join(dev_dir, '外部验证',
                               f'{val_key}_external_validation_report.txt'),
                  'w', encoding='utf-8') as f:
            f.write(f"推导: {dev_key} | 验证: {val_key} | 模型: {best_name}\n")
            f.write(f"n={len(y_v)}, Acc={v_acc:.4f}, MacroAUC={v_auc:.4f}\n")
            for i, cn in enumerate(['non-Flu', 'FluA', 'FluB']):
                f.write(f"{cn} AUC: {v_aucs[i]:.4f}\n")

# ============ 汇总 eFigure 1 数据 ============
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'eFigure1_summary.csv'),
                  index=False, encoding='utf-8-sig')
print("\n" + "=" * 60)
print(summary_df.pivot_table(index='Derivation', columns='Validation',
                             values='MacroAUC').to_string())

# --- eFigure 1 热图 ---
pivot_auc = summary_df.pivot_table(index='Derivation', columns='Validation',
                                   values='MacroAUC')
pivot_acc = summary_df.pivot_table(index='Derivation', columns='Validation',
                                   values='Accuracy')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, pv, title in [(axes[0], pivot_acc, 'Accuracy'),
                      (axes[1], pivot_auc, 'Macro AUC')]:
    sns_im = ax.imshow(pv.values, cmap='RdYlGn', vmin=0.4, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns)
    ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=9)
    ax.set_title(f'{title} by Derivation Cohort', fontweight='bold')
    ax.set_xlabel('Validation Cohort'); ax.set_ylabel('Derivation Cohort')
    plt.colorbar(sns_im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'eFigure1_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n📁 结果已保存至: {OUTPUT_DIR}")

# -*- coding: utf-8 -*-
"""
02_模型选择_六模型网格搜索.py
============================================================
功能：附三院（TAH）推导队列上，六模型网格搜索 + 内部验证（70:30分层拆分）
      + Bootstrap 1000次 95% CI → 对应文章 eTable 5 / Table 2
模型：RandomForest / LightGBM / XGBoost / LogisticRegression / AdaBoost / SVM
      (与原 模型选择.txt 一致，参数网格、random_state=42 均未改动)

输出：代码存档/运行结果/02_模型构建/
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (roc_curve, auc, classification_report,
                             confusion_matrix, accuracy_score)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置区 ====================
BASE_DIR = os.environ.get("INFLUENZA_DATA_DIR", r"E:\上班后的资料\甲乙流多中心3")
MODELING_DATA = os.path.join(BASE_DIR, 'a附三院数据_processed.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, r'代码存档\运行结果\02_模型构建')
RANDOM_STATE = 42
N_BOOTSTRAPS = 1000
# ================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
class_names = {0: 'non-Flu', 1: 'FluA', 2: 'FluB'}
target_names = [class_names[i] for i in sorted(class_names)]

df_modeling = pd.read_csv(MODELING_DATA, encoding='utf-8-sig')
X, y = df_modeling.drop('RESULT', axis=1), df_modeling['RESULT']

# --- 变量类型：sex 强制分类，month 按数值 Z 标准化（与原代码一致）---
all_numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
forced_categorical_cols = [c for c in ['sex'] if c in X.columns]
object_categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
final_categorical_cols = list(set(object_categorical_cols + forced_categorical_cols))
final_numeric_cols = [c for c in all_numeric_cols if c not in final_categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), final_numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), final_categorical_cols)
    ],
    remainder='passthrough')

# --- 70:30 分层拆分 ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)
print(f"训练集 {len(X_train):,} 行 / 验证集 {len(X_val):,} 行")

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

# --- 模型与超参网格（与原代码一致）---
model_configs = {
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {'n_estimators': [100, 200], 'max_depth': [10, 20],
                   'min_samples_leaf': [1, 4]}},
    "LightGBM": {
        "model": LGBMClassifier(random_state=42, verbose=-1) if LGBMClassifier else None,
        "params": {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1],
                   'num_leaves': [31, 50]}},
    "XGBoost": {
        "model": XGBClassifier(random_state=42, eval_metric='logloss') if XGBClassifier else None,
        "params": {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1],
                   'max_depth': [3, 6]}},
    "LogisticRegression": {
        "model": LogisticRegression(random_state=42, max_iter=1000),
        "params": {'C': [0.1, 1, 10]}},
    "AdaBoost": {
        "model": AdaBoostClassifier(random_state=42),
        "params": {'n_estimators': [50, 100], 'learning_rate': [0.5, 1.0]}},
    "SVM": {
        "model": SVC(probability=True, random_state=42),
        "params": {'C': [1, 10], 'kernel': ['rbf', 'linear']}},
}


def compute_bootstrap_ci(y_true, y_proba, n_bootstraps=1000, rng=None):
    """Bootstrap 计算宏平均 ROC AUC 的 95% CI（Numpy 数组版，避免索引问题）"""
    if rng is None:
        rng = np.random.RandomState(RANDOM_STATE)
    y_true_arr, y_proba_arr = np.array(y_true), np.array(y_proba)
    scores, tprs_all = [], []
    base_fpr = np.linspace(0, 1, 100)
    n_classes = y_proba_arr.shape[1]
    n_samples = len(y_true_arr)

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n_samples, n_samples)
        yb, pb = y_true_arr[idx], y_proba_arr[idx]
        if len(np.unique(yb)) < 2:
            continue
        fpr_d, tpr_d, auc_d = {}, {}, {}
        ok = True
        for c in range(n_classes):
            try:
                fpr, tpr, _ = roc_curve((yb == c).astype(int), pb[:, c])
                auc_d[c] = auc(fpr, tpr)
                itpr = np.interp(base_fpr, fpr, tpr)
                itpr[0] = 0.0
                tpr_d[c] = itpr
            except Exception:
                ok = False
                break
        if ok and len(auc_d) == n_classes:
            mean_tpr = np.mean([tpr_d[c] for c in range(n_classes)], axis=0)
            scores.append(np.mean(list(auc_d.values())))
            tprs_all.append(mean_tpr)

    scores = np.array(scores)
    tprs_all = np.array(tprs_all)
    return (base_fpr,
            np.percentile(tprs_all, 2.5, axis=0),
            np.percentile(tprs_all, 97.5, axis=0),
            scores.mean(),
            np.percentile(scores, 2.5),
            np.percentile(scores, 97.5))


# --- 训练与评估 ---
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for idx, (name, config) in enumerate(model_configs.items(), 1):
    print(f"\n[{idx}/6] 训练 {name}")
    if config["model"] is None:
        print(f"  跳过 {name}（库未安装）")
        continue
    gs = GridSearchCV(config["model"], config["params"], cv=cv,
                      scoring='roc_auc_ovr', n_jobs=-1)
    gs.fit(X_train_processed, y_train)
    best_model = gs.best_estimator_

    y_pred = best_model.predict(X_val_processed)
    y_proba = best_model.predict_proba(X_val_processed)

    fpr, tpr, roc_auc = {}, {}, {}
    n_classes = 3
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_val, y_proba[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(all_fpr, mean_tpr)

    _, tpr_lo, tpr_hi, auc_mean, auc_lo, auc_hi = compute_bootstrap_ci(
        y_val, y_proba, n_bootstraps=N_BOOTSTRAPS)

    results[name] = {
        "best_model": best_model,
        "best_params": gs.best_params_,
        "best_cv_score": gs.best_score_,
        "accuracy": accuracy_score(y_val, y_pred),
        "classification_report": classification_report(y_val, y_pred, target_names=target_names),
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_val, y_pred),
        "auc_ci": (auc_lo, auc_hi)}
    print(f"  最佳参数: {gs.best_params_}")
    print(f"  宏平均AUC: {roc_auc['macro']:.4f} (95% CI: {auc_lo:.4f}-{auc_hi:.4f})")

# --- 保存最佳模型 ---
best_model_name = max(results, key=lambda k: results[k]['roc_auc']['macro'])
with open(os.path.join(OUTPUT_DIR, 'best_model.pkl'), 'wb') as f:
    pickle.dump(results[best_model_name]['best_model'], f)
with open(os.path.join(OUTPUT_DIR, 'preprocessor.pkl'), 'wb') as f:
    pickle.dump(preprocessor, f)
with open(os.path.join(OUTPUT_DIR, 'feature_names.pkl'), 'wb') as f:
    pickle.dump(list(X.columns), f)

# --- 可视化：ROC对比（带CI）+ 性能排名表 ---
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
linestyles = ['-', '--', '-.', ':', '-', '--']
plt.figure(figsize=(12, 8))
legend_entries = []
for i, (name, res) in enumerate(results.items()):
    plt.plot(res["fpr"]["macro"], res["tpr"]["macro"],
             color=colors[i % 6], linestyle=linestyles[i % 6], lw=2)
    lo, hi = res["auc_ci"]
    legend_entries.append(
        f'{name} (AUC={res["roc_auc"]["macro"]:.3f}, 95%CI:{lo:.3f}-{hi:.3f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0, 1]); plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Model Comparison: Macro-average ROC Curves with 95% CI',
          fontsize=14, fontweight='bold')
plt.legend(legend_entries, loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'macro_roc_comparison_with_CI.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# --- 详细报告 ---
with open(os.path.join(OUTPUT_DIR, 'model_evaluation_report_with_CI.txt'), 'w',
          encoding='utf-8') as f:
    f.write("附三院（TAH）推导队列 六模型评估报告 (含 95% CI)\n" + "=" * 50 + "\n\n")
    f.write("=== 模型性能排名 ===\n")
    for rank, (name, res) in enumerate(
            sorted(results.items(), key=lambda x: x[1]['roc_auc']['macro'], reverse=True), 1):
        lo, hi = res['auc_ci']
        f.write(f"{rank}. {name}: Macro AUC = {res['roc_auc']['macro']:.4f} "
                f"(95% CI: {lo:.4f}-{hi:.4f})\n")
    f.write("\n")
    for name, res in results.items():
        lo, hi = res['auc_ci']
        f.write(f"\n{'='*20} {name} {'='*20}\n")
        f.write(f"最佳参数: {res['best_params']}\n")
        f.write(f"最佳交叉验证AUC: {res['best_cv_score']:.4f}\n")
        f.write(f"验证集准确率: {res['accuracy']:.4f}\n")
        f.write(f"宏平均AUC: {res['roc_auc']['macro']:.4f} (95% CI: {lo:.4f}-{hi:.4f})\n\n")
        f.write(res['classification_report'] + "\n")

print(f"\n✅ 最佳模型: {best_model_name}")
print(f"📁 结果已保存至: {OUTPUT_DIR}")

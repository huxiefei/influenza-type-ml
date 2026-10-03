# -*- coding: utf-8 -*-
"""
07_SHAP分析.py  【原脚本散佚，本脚本为按附三9产物格式重写的完整复现版】
============================================================
功能（三部分，对应文章 Figure 3/4 与补充材料 SHAP 图）：
  Part 1  各中心分类别 SHAP：TAH/SAH/CEMC/FDH × non-Flu/FluA/FluB
          → 01_SHAP_Summary_{类别}.png（beeswarm）
          → feature_importance_{类别}.csv（内置gain + 置换重要性 + 排名）
          → feature_importance_by_class.png
  Part 2  三向鉴别分析（TAH推导队列）：
          → 01_Discriminative_Power_Ranking.png  三类鉴别力排名
          → 02_Top5_Contribution_Comparison.png   Top5特征三类贡献对比
          → 03_No1_Feature_Distribution.png       头号特征分布
          → Top5_Dependence_{特征}.png            Top5依赖图
          → Discrimination_Score_Table.xlsx
  Part 3  策略一多中心验证（SAH+CEMC+FDH合并，FluA）：
          → 01_Global_Beeswarm_FluA / 02_Global_Bar_Top15_FluA
          → 03_Center_Comparison_Bar_FluA / 04_Feature_Ranking_Table_FluA.xlsx

输出：代码存档/运行结果/07_SHAP分析/
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # 无界面环境，避免 Qt 后端
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置区 ====================
BASE_DIR = os.environ.get("INFLUENZA_DATA_DIR", r"E:\上班后的资料\甲乙流多中心3")
MODEL_DIR = os.path.join(BASE_DIR, r'代码存档\运行结果\03_最佳模型')
if not os.path.exists(os.path.join(MODEL_DIR, 'final_lightgbm_model.pkl')):
    MODEL_DIR = os.path.join(BASE_DIR, r'附三8\最佳模型')   # 回退到原始模型
OUTPUT_DIR = os.path.join(BASE_DIR, r'代码存档\运行结果\07_SHAP分析')
RANDOM_STATE = 42
N_REPEATS = 10
CLASS_NAME = {0: 'non-Flu', 1: 'FluA', 2: 'FluB'}

CENTER_DATA = {
    'TAH':  ('附三中心', os.path.join(BASE_DIR, 'a附三院数据_processed.csv')),
    'SAH':  ('新桥中心', os.path.join(BASE_DIR, 'b新桥数据_processed.csv')),
    'CEMC': ('急救中心', os.path.join(BASE_DIR, 'c急救数据.csv')),
    'FDH':  ('丰都中心', os.path.join(BASE_DIR, 'd丰都数据_processed.csv')),
}
# ================================================

# --- 加载模型三件套 ---
with open(os.path.join(MODEL_DIR, 'final_lightgbm_model.pkl'), 'rb') as f:
    final_model = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'preprocessor.pkl'), 'rb') as f:
    preprocessor = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'feature_names.pkl'), 'rb') as f:
    feat_names = pickle.load(f)

explainer = shap.TreeExplainer(final_model)
proc_feat_names = list(preprocessor.get_feature_names_out())


def clean_name(n):
    """去掉 ColumnTransformer 前缀，用于绘图展示（num__LYM → LYM）"""
    if n.startswith('num__'):
        return n[5:]
    if n.startswith('cat__'):
        return n[5:].split('_')[0]
    return n


disp_names = [clean_name(n) for n in proc_feat_names]


def load_center_processed(key):
    """读取中心数据并做预处理，返回 (X_processed DataFrame, y)"""
    _, path = CENTER_DATA[key]
    df = pd.read_csv(path, encoding='utf-8-sig')
    y = df['RESULT']
    X = df.drop('RESULT', axis=1)
    for c in feat_names:
        if c not in X.columns:
            X[c] = 0
    X = X[feat_names]
    X_p = preprocessor.transform(X)
    return pd.DataFrame(X_p, columns=proc_feat_names), y


def get_shap_values(X_p_df):
    """TreeExplainer SHAP 值，统一为 [n_samples, n_features, n_classes]"""
    sv = explainer.shap_values(X_p_df, check_additivity=False)
    if isinstance(sv, list):          # 旧版 shap: list(3) of [n, f]
        sv = np.stack(sv, axis=-1)
    if sv.ndim == 3 and sv.shape[-1] != 3:
        sv = np.transpose(sv, (1, 2, 0))  # [c, n, f] → [n, f, c]
    return sv


# ============================================================
# Part 1: 各中心分类别 SHAP
# ============================================================
print("=" * 60)
print("Part 1: 各中心分类别 SHAP 分析")
print("=" * 60)

all_center_mean_shap = {}   # 供 Part 3 使用

for key in ['TAH', 'SAH', 'CEMC', 'FDH']:
    cn = CENTER_DATA[key][0]
    out_dir = os.path.join(OUTPUT_DIR, f'SHAP分析_{cn}_分类别')
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n--- {cn} ({key}) ---")

    X_p, y = load_center_processed(key)
    sv = get_shap_values(X_p)
    all_center_mean_shap[key] = {c: np.abs(sv[:, :, c]).mean(axis=0) for c in range(3)}

    X_disp = X_p.copy()
    X_disp.columns = disp_names      # 展示用清洁特征名

    for c in range(3):
        cname = CLASS_NAME[c]

        # 1a. beeswarm 汇总图
        plt.figure()
        shap.summary_plot(sv[:, :, c], X_disp, show=False, max_display=20)
        plt.title(f'SHAP Summary for {cname} ({key})', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'01_SHAP_Summary_{cname}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 1b. 内置 gain + 置换重要性（该类别 OvR AUC 记分）
        internal_imp = final_model.feature_importances_

        def _cls_auc_scorer(est, X, yy, cls=c):
            probs = est.predict_proba(X)
            return roc_auc_score((yy == cls).astype(int), probs[:, cls])

        perm = permutation_importance(final_model, X_p.values, y,
                                      scoring=_cls_auc_scorer,
                                      n_repeats=N_REPEATS,
                                      random_state=RANDOM_STATE, n_jobs=-1)
        imp_df = pd.DataFrame({
            'Feature': proc_feat_names,
            'Internal_importance': internal_imp,
            'Perm_importance_mean': perm.importances_mean,
            'Perm_importance_std': perm.importances_std})
        imp_df['Internal_rank'] = imp_df['Internal_importance'].rank(
            ascending=False).astype(int)
        imp_df['Perm_rank'] = imp_df['Perm_importance_mean'].rank(
            ascending=False).astype(int)
        imp_df = imp_df.sort_values('Internal_importance', ascending=False)
        imp_df.to_csv(os.path.join(out_dir, f'feature_importance_{cname}.csv'),
                      index=False, encoding='utf-8-sig')
        print(f"  ✅ {cname}: summary图 + importance CSV 已保存")

    # 1c. 三类别重要性对比图
    plt.figure(figsize=(12, 8))
    top_feats = imp_df['Feature'].head(15).tolist()
    x = np.arange(len(top_feats))
    w = 0.27
    for c in range(3):
        vals = [all_center_mean_shap[key][c][proc_feat_names.index(f)] for f in top_feats]
        plt.bar(x + (c - 1) * w, vals, w, label=CLASS_NAME[c])
    plt.xticks(x, top_feats, rotation=45, ha='right', fontsize=8)
    plt.ylabel('Mean |SHAP value|')
    plt.title(f'Feature Importance by Class — {key}', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'feature_importance_by_class.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================
# Part 2: 三向鉴别分析（TAH 推导队列）
# ============================================================
print("\n" + "=" * 60)
print("Part 2: 三向鉴别分析 (TAH)")
print("=" * 60)

out_dir = os.path.join(OUTPUT_DIR, 'SHAP分析_三向鉴别')
os.makedirs(out_dir, exist_ok=True)

X_tah, y_tah = load_center_processed('TAH')
sv_tah = get_shap_values(X_tah)

# 2a. 鉴别力得分 = 各类别平均|SHAP|的离散度（max-min）
mean_abs = np.stack([np.abs(sv_tah[:, :, c]).mean(axis=0) for c in range(3)])  # [3, f]
disc_score = mean_abs.max(axis=0) - mean_abs.min(axis=0)
disc_df = pd.DataFrame({'Feature': proc_feat_names, 'Discrimination_Score': disc_score,
                        'SHAP_nonFlu': mean_abs[0], 'SHAP_FluA': mean_abs[1],
                        'SHAP_FluB': mean_abs[2]}
                       ).sort_values('Discrimination_Score', ascending=False)
disc_df.to_excel(os.path.join(out_dir, 'Discrimination_Score_Table.xlsx'), index=False)

top15 = disc_df.head(15)
plt.figure(figsize=(10, 8))
plt.barh([f.replace('num__', '').replace('cat__', '')
          for f in top15['Feature']][::-1],
         top15['Discrimination_Score'][::-1],
         color='steelblue', alpha=0.85)
plt.xlabel('Discrimination Score (max − min of class-wise mean |SHAP|)')
plt.title('Discriminative Power Ranking (TAH, three-way)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '01_Discriminative_Power_Ranking.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# 2b. Top5 特征三类贡献对比
top5 = disc_df.head(5)
x = np.arange(len(top5))
plt.figure(figsize=(10, 6))
w = 0.25
for c in range(3):
    plt.bar(x + (c - 1) * w,
            [mean_abs[c][proc_feat_names.index(f)] for f in top5['Feature']], w,
            label=CLASS_NAME[c])
plt.xticks(x, top5['Feature'], rotation=30, ha='right')
plt.ylabel('Mean |SHAP value|')
plt.title('Top-5 Features: Class-wise Contribution Comparison', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '02_Top5_Contribution_Comparison.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# 2c. 头号特征分布
no1_feat = disc_df.iloc[0]['Feature']
col_idx = proc_feat_names.index(no1_feat)
plt.figure(figsize=(10, 6))
for c in range(3):
    mask = (y_tah.values == c).values if hasattr(y_tah.values == c, 'values') else (y_tah.values == c)
    plt.hist(X_tah.iloc[np.where(mask)[0], col_idx], bins=40, alpha=0.55,
             density=True, label=CLASS_NAME[c])
plt.xlabel(f'{no1_feat} (standardized value)')
plt.ylabel('Density')
plt.title(f'Distribution of No.1 Feature: {no1_feat} by Class', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '03_No1_Feature_Distribution.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# 2d. Top5 依赖图（SHAP vs 特征值，按类别着色）
for feat in top5['Feature']:
    ci = proc_feat_names.index(feat)
    plt.figure(figsize=(8, 6))
    y_arr = y_tah.values if hasattr(y_tah, 'values') else np.asarray(y_tah)
    sc = plt.scatter(X_tah.iloc[:, ci], sv_tah[:, ci, 1], c=y_arr,
                     cmap='coolwarm', alpha=0.4, s=12)
    plt.colorbar(sc, ticks=[0, 1, 2], label='Class')
    plt.axhline(0, color='gray', linestyle='--', lw=0.8)
    plt.xlabel(f'{feat.replace("num__", "")} (standardized)')
    plt.ylabel('SHAP value (FluA)')
    plt.title(f'Dependence: {feat.replace("num__", "")}', fontweight='bold')
    plt.tight_layout()
    safe = feat.replace('%', 'pct').replace('num__', '').replace('cat__', '')
    plt.savefig(os.path.join(out_dir, f'Top5_Dependence_{safe}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
print("✅ 三向鉴别分析完成")

# ============================================================
# Part 3: 策略一多中心验证（SAH+CEMC+FDH 合并，FluA）
# ============================================================
print("\n" + "=" * 60)
print("Part 3: 策略一多中心验证 (SAH+CEMC+FDH, FluA)")
print("=" * 60)

out_dir = os.path.join(OUTPUT_DIR, 'SHAP分析_策略一多中心验证')
os.makedirs(out_dir, exist_ok=True)

X_pool, y_pool = [], []
for key in ['SAH', 'CEMC', 'FDH']:
    X_p, y = load_center_processed(key)
    X_pool.append(X_p)
    y_pool.append(y)
X_pool_df = pd.concat(X_pool, ignore_index=True)
y_pool_arr = np.concatenate([np.asarray(v) for v in y_pool])

sv_pool = get_shap_values(X_pool_df)
cls = 1  # FluA
X_pool_disp = X_pool_df.copy()
X_pool_disp.columns = disp_names

# 3a. 全局 beeswarm
plt.figure()
shap.summary_plot(sv_pool[:, :, cls], X_pool_disp, show=False, max_display=20)
plt.title('Global SHAP Beeswarm — FluA (pooled external centers)', fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '01_Global_Beeswarm_FluA.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# 3b. 全局 Top15 bar
mean_abs_pool = np.abs(sv_pool[:, :, cls]).mean(axis=0)
order = np.argsort(mean_abs_pool)[::-1][:15]
plt.figure(figsize=(10, 7))
plt.barh([disp_names[i] for i in order][::-1],
         mean_abs_pool[order][::-1], color='steelblue')
plt.xlabel('Mean |SHAP value|')
plt.title('Global Feature Importance (Top 15) — FluA', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '02_Global_Bar_Top15_FluA.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# 3c. 三中心对比 bar（各中心平均|SHAP| Top12）
top12 = order[:12]
x = np.arange(len(top12))
plt.figure(figsize=(12, 6))
w = 0.25
center_colors = {'SAH': '#1f77b4', 'CEMC': '#ff7f0e', 'FDH': '#2ca02c'}
for i, key in enumerate(['SAH', 'CEMC', 'FDH']):
    vals = [all_center_mean_shap[key][cls][j] for j in top12]
    plt.bar(x + (i - 1) * w, vals, w, label=key, color=center_colors[key])
plt.xticks(x, [disp_names[j] for j in top12], rotation=45, ha='right', fontsize=8)
plt.ylabel('Mean |SHAP value| (FluA)')
plt.title('Feature Importance Comparison Across External Centers — FluA',
          fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '03_Center_Comparison_Bar_FluA.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# 3d. 特征排名表
rank_df = pd.DataFrame({'Feature': proc_feat_names,
                        'MeanAbsSHAP_FluA_pooled': mean_abs_pool,
                        'MeanAbsSHAP_FluA_SAH': [all_center_mean_shap['SAH'][cls][j]
                                                 for j in range(len(proc_feat_names))],
                        'MeanAbsSHAP_FluA_CEMC': [all_center_mean_shap['CEMC'][cls][j]
                                                  for j in range(len(proc_feat_names))],
                        'MeanAbsSHAP_FluA_FDH': [all_center_mean_shap['FDH'][cls][j]
                                                 for j in range(len(proc_feat_names))]
                        }).sort_values('MeanAbsSHAP_FluA_pooled', ascending=False)
rank_df.to_excel(os.path.join(out_dir, '04_Feature_Ranking_Table_FluA.xlsx'), index=False)

print(f"\n📁 全部 SHAP 结果已保存至: {OUTPUT_DIR}")
print("完成。请将生成图与附三9原始图对比核验样式后用于投稿。")

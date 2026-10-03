# -*- coding: utf-8 -*-
"""
01_数据预处理_KNN填补_3SD异常值处理.py
============================================================
功能：
    对四个中心的原始数据（年龄/性别/月份 + 血常规25项）进行
      1) KNN 填补缺失值
      2) 均值 ± 3SD 异常值缩尾处理（超界值替换为边界值）
    生成建模用的 *_processed.csv

说明：
    原预处理脚本已散佚，本脚本按文章 Methods 描述与
    processed_data_fixed/processing_summary.txt（2025-12-15版）重写。
    【可调假设】KNN_K=5、SD_TIMES=3、各中心独立处理（互不借用信息），
    如与当时的实际参数不同，请只改本文件顶部配置区。

输出：
    代码存档/运行结果/01_预处理/
        a附三院数据_processed.csv / b新桥数据_processed.csv
        c急救数据_processed.csv  / d丰都数据_processed.csv
        preprocessing_summary.txt
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.impute import KNNImputer

# ==================== 配置区 ====================
BASE_DIR = os.environ.get("INFLUENZA_DATA_DIR", r"E:\上班后的资料\甲乙流多中心3")
OUTPUT_DIR = os.path.join(BASE_DIR, r'代码存档\运行结果\01_预处理')

# 原始数据（含缺失值的原始导出文件）。请按实际情况修改路径。
# 若原始文件不在以下位置，请把四个原始csv放到 代码存档/原始数据/ 文件夹内。
RAW_DIR = os.path.join(BASE_DIR, r'代码存档\原始数据')

INPUT_FILES = {
    '附三院数据': 'a附三院原始数据.csv',   # → TAH 推导队列
    '新桥数据':   'b新桥原始数据.csv',     # → SAH 外部验证
    '急救数据':   'c急救原始数据.csv',     # → CEMC 外部验证
    '丰都数据':   'd丰都原始数据.csv',     # → FDH 外部验证
}
OUTPUT_SUFFIX = {
    '附三院数据': 'a附三院数据_processed.csv',
    '新桥数据':   'b新桥数据_processed.csv',
    '急救数据':   'c急救数据_processed.csv',
    '丰都数据':   'd丰都数据_processed.csv',
}

# 建模所需的26列（25特征 + RESULT）
REQUIRED_COLS = ['age', 'sex', 'month', 'WBC', 'NEU', 'NEU%', 'LYM', 'LYM%',
                 'BASO', 'BASO%', 'EO', 'EO%', 'MON', 'MON%', 'RBC', 'HGB',
                 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT', 'MPV', 'PCT', 'PDW',
                 'RDW-CV', 'RESULT']

KNN_K = 5          # KNN 填补近邻数
SD_TIMES = 3       # 异常值判定倍数（均值 ± 3SD）
RANDOM_STATE = 42
# ================================================


def preprocess_one(df: pd.DataFrame, name: str, log: list) -> pd.DataFrame:
    """单个中心：列对齐 → KNN填补 → 3SD缩尾"""
    n0 = len(df)

    # --- 列对齐（缺少的列补 NaN，多余的列丢弃）---
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[REQUIRED_COLS].copy()

    # 删除标签缺失的行
    df = df.dropna(subset=['RESULT'])
    df['RESULT'] = df['RESULT'].astype(int)

    # --- KNN 填补（数值特征，含 sex/month；各中心独立拟合）---
    feature_cols = [c for c in REQUIRED_COLS if c != 'RESULT']
    n_missing_before = int(df[feature_cols].isna().sum().sum())

    imputer = KNNImputer(n_neighbors=KNN_K)
    df[feature_cols] = imputer.fit_transform(df[feature_cols])

    # sex 二值化、month 取整
    df['sex'] = (df['sex'] >= 0.5).astype(float)
    df['month'] = df['month'].round().clip(1, 12).astype(float)

    # --- 3SD 异常值缩尾（age/month/sex 不处理，各中心独立计算）---
    winsor_cols = [c for c in feature_cols if c not in ('sex', 'month')]
    n_outlier = 0
    for col in winsor_cols:
        mu, sd = df[col].mean(), df[col].std()
        if sd == 0 or np.isnan(sd):
            continue
        lower, upper = mu - SD_TIMES * sd, mu + SD_TIMES * sd
        mask = (df[col] < lower) | (df[col] > upper)
        n_outlier += int(mask.sum())
        df[col] = df[col].clip(lower, upper)

    log.append(f"数据集: {name}")
    log.append(f"  原始行数: {n0} → 有效行数: {len(df)}")
    log.append(f"  KNN填补缺失值数: {n_missing_before} (k={KNN_K})")
    log.append(f"  {SD_TIMES}SD缩尾处理特征数: {len(winsor_cols)}, 替换值数: {n_outlier}")
    log.append(f"  剩余缺失值: {int(df.isna().sum().sum())}")
    log.append(f"  RESULT分布: {df['RESULT'].value_counts().to_dict()}")
    log.append("")
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log = [f"数据预处理汇总报告（重写版）",
           f"处理时间: {datetime.now()}",
           f"KNN_K={KNN_K}, SD_TIMES={SD_TIMES}, 各中心独立处理",
           "=" * 50, ""]

    for name, fname in INPUT_FILES.items():
        path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(path):
            log.append(f"数据集: {name} —— ⚠️ 未找到原始文件 {fname}，跳过")
            log.append("")
            continue
        df = pd.read_csv(path, encoding='utf-8-sig')
        df_out = preprocess_one(df, name, log)
        out_path = os.path.join(OUTPUT_DIR, OUTPUT_SUFFIX[name])
        df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"✅ {name}: {df_out.shape} → {out_path}")

    with open(os.path.join(OUTPUT_DIR, 'preprocessing_summary.txt'), 'w',
              encoding='utf-8') as f:
        f.write("\n".join(log))
    print(f"✅ 报告已保存: {os.path.join(OUTPUT_DIR, 'preprocessing_summary.txt')}")


if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np

df = pd.read_csv("q1_daily_merged.csv", parse_dates=["date"]).sort_values("date")

# 1) 原计划的气象列
met_cols = ["T","U","Ff","ff10","VV","Td","VPD","RRR","Po","P","Pa","T_max","T_min"]

# 2) 剔除全为空的列（你这份里 ff10 就会被剔除）
full_nan = [c for c in met_cols if c not in df.columns or df[c].isna().all()]
met_cols = [c for c in met_cols if c not in full_nan]
print("已剔除全空列:", full_nan)

# 3) 对剩余列做简单填补（先时序前向/后向，再用中位数兜底）
df[met_cols] = df[met_cols].ffill().bfill()
for c in met_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())

# 4) 造滞后与累计特征（严格使用“过去信息”）
for c in met_cols:
    df[c + "_lag1"] = df[c].shift(1)          # 昨日

# 降水累计（把 NaN 当 0 处理，再做 rolling）
rrr = df["RRR"].fillna(0) if "RRR" in df.columns else pd.Series(0, index=df.index)
df["RRR_sum3"] = rrr.rolling(3, min_periods=1).sum().shift(1)   # 过去3天
df["RRR_sum7"] = rrr.rolling(7, min_periods=1).sum().shift(1)   # 过去7天

# 可选：加入日气压变化（避免泄漏：先做diff，再整体滞后1天）
for c in ["P","Po"]:
    if c in df.columns:
        df[c+"_diff1_lag1"] = (df[c] - df[c].shift(1)).shift(1)

# 5) 组装训练集
feats = [c + "_lag1" for c in met_cols] + ["RRR_sum3","RRR_sum7"] \
        + [c+"_diff1_lag1" for c in ["P","Po"] if c in df.columns]
data = df.dropna(subset=feats + ["sm5"]).copy()

print("样本量：", len(data))
print("特征数：", len(feats))
print("前几列预览：")
print(data[["date","sm5"] + feats[:5]].head())

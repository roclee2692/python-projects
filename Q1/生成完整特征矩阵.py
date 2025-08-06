import pandas as pd, numpy as np, pyarrow as pa, pyarrow.parquet as pq

df = pd.read_csv(r"D:\python-projects\q1_daily_merged.csv",
                 parse_dates=["date"]).sort_values("date")

# === 造特征（必须跟训练脚本一模一样）=====================
met_cols = ["T","U","Ff","VV","Td","VPD","RRR","Po","P","Pa","T_max","T_min"]
for c in met_cols:
    df[c+"_lag1"] = df[c].shift(1)
df["RRR_sum3"] = df["RRR"].fillna(0).rolling(3, min_periods=1).sum().shift(1)
df["RRR_sum7"] = df["RRR"].fillna(0).rolling(7, min_periods=1).sum().shift(1)
for c in ["P","Po"]:
    df[c+"_diff1_lag1"] = (df[c] - df[c].shift(1)).shift(1)
# ----------------------------------------------

feats  = [c+"_lag1" for c in met_cols] + ["RRR_sum3","RRR_sum7"] \
         + [c+"_diff1_lag1" for c in ["P","Po"]]
target = "sm5"                          # <——🔵 你的标签列名

Xy = df.dropna(subset=feats+[target]).copy()
pq.write_table(pa.Table.from_pandas(Xy[feats+[target]]),
               r"D:\python-projects\q1_dataset.parquet")
print("✅ 已导出 q1_dataset.parquet：", Xy.shape)

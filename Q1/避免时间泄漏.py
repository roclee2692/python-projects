import pandas as pd
df = pd.read_csv("q1_daily_merged.csv", parse_dates=["date"]).sort_values("date")

met_cols = ["T","U","Ff","ff10","VV","Td","VPD","RRR","Po","P","Pa","T_max","T_min"]
for c in met_cols:
    df[c+"_lag1"] = df[c].shift(1)               # 昨日特征

df["RRR_sum3"] = df["RRR"].rolling(3, min_periods=1).sum().shift(1)  # 过去3天降水
df["RRR_sum7"] = df["RRR"].rolling(7, min_periods=1).sum().shift(1)  # 过去7天降水

# 选择训练列
feats = [c+"_lag1" for c in met_cols] + ["RRR_sum3","RRR_sum7"]
data = df.dropna(subset=feats+["sm5"]).copy()
df = pd.read_csv("q1_daily_merged.csv")
print(df.isna().mean().sort_values(ascending=False).head(10))
print((df["obs48"]<40).sum())

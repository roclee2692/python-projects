# -*- coding: utf-8 -*-
import json, joblib, warnings, os
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ========= 路径配置（按你机器改 IN_CSV 即可）=========
IN_CSV  = r"D:\python-projects\q1_daily_merged.csv"
OUT_DIR = Path(r"D:\python-projects")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_P = OUT_DIR / "model.pkl"
FEATS_P = OUT_DIR / "feature_list.json"
RPT_P   = OUT_DIR / "train_report.json"
PREV_P  = OUT_DIR / "train_preview.csv"

warnings.filterwarnings("ignore")

# 读取
df = pd.read_csv(IN_CSV, parse_dates=["date"]).sort_values("date")
print(f"✅ 读取合并数据：{len(df)} 天，列数 {len(df.columns)}")

# 剔除全空列 & 填补
met_cols = ["T","U","Ff","ff10","VV","Td","VPD","RRR","Po","P","Pa","T_max","T_min"]
full_nan = [c for c in met_cols if c not in df.columns or df[c].isna().all()]
met_cols = [c for c in met_cols if c not in full_nan]
print("🧹 已剔除全空列：", full_nan)

df[met_cols] = df[met_cols].ffill().bfill()
for c in met_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())

# 不泄漏的滞后/累计
for c in met_cols:
    df[c+"_lag1"] = df[c].shift(1)
rrr = df["RRR"].fillna(0) if "RRR" in df.columns else pd.Series(0, index=df.index)
df["RRR_sum3"] = rrr.rolling(3, min_periods=1).sum().shift(1)
df["RRR_sum7"] = rrr.rolling(7, min_periods=1).sum().shift(1)
for c in ["P","Po"]:
    if c in df.columns:
        df[c+"_diff1_lag1"] = (df[c]-df[c].shift(1)).shift(1)

feats = [c+"_lag1" for c in met_cols] + ["RRR_sum3","RRR_sum7"] \
        + [c+"_diff1_lag1" for c in ["P","Po"] if c in df.columns]
target = "sm5"

data = df.dropna(subset=feats+[target]).copy()
print(f"📦 可训练样本：{len(data)}，特征数：{len(feats)}")

# 时序划分
split = int(len(data)*0.8)
tr, te = data.iloc[:split], data.iloc[split:]
Xtr, ytr = tr[feats], tr[target]
Xte, yte = te[feats], te[target]

# 模型：优先 LightGBM，不行就 RF
model_name = ""
try:
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=127,
        min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=1, random_state=42, n_jobs=-1
    )
    # 关键：不传 verbose；用 callbacks 控制早停/日志
    callbacks = [lgb.early_stopping(100), lgb.log_evaluation(0)]
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)], eval_metric="l2", callbacks=callbacks)
    model_name = "LightGBM"
except Exception as e:
    print("⚠️ LightGBM 不可用，退化为随机森林：", e)
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1)
    model.fit(Xtr, ytr)
    model_name = "RandomForest"

# 评估（兼容老版本 sklearn：RMSE 自己开方）
pred = model.predict(Xte)
mae  = mean_absolute_error(yte, pred)
mse  = mean_squared_error(yte, pred)          # 不用 squared=False
rmse = float(np.sqrt(mse))
r2   = r2_score(yte, pred)
print(f"📊 测试集指标 | MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  模型={model_name}")

# 导出
import joblib, json
joblib.dump(model, MODEL_P)
FEATS_P.write_text(json.dumps({"feats":feats}, ensure_ascii=False, indent=2), encoding="utf-8")
data[["date",target]+feats].head(20).to_csv(PREV_P, index=False, encoding="utf-8-sig")
RPT_P.write_text(json.dumps({
    "rows_total": int(len(df)),
    "rows_trainable": int(len(data)),
    "split_train": int(len(Xtr)),
    "split_test": int(len(Xte)),
    "features": feats,
    "removed_full_nan": full_nan,
    "metrics": {"MAE":mae, "RMSE":rmse, "R2":r2},
    "model": model_name
}, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n🎉 训练完成：")
print(f"- 模型：{MODEL_P}")
print(f"- 特征清单：{FEATS_P}")
print(f"- 指标报告：{RPT_P}")
print(f"- 预览数据：{PREV_P}")

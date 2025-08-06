# -*- coding: utf-8 -*-
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUT_DIR = Path(r"D:\python-projects")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) 自动挑选预测文件
cand = [OUT_DIR/"lgbm_pred.csv", OUT_DIR/"final_pred.csv"]
pred_path = next((str(p) for p in cand if p.exists()), None)
if pred_path is None:
    raise FileNotFoundError("找不到预测文件：期待 D:\\python-projects\\lgbm_pred.csv 或 final_pred.csv")

print("✅ 使用预测文件：", pred_path)

# 2) 读取并对齐列名
df = pd.read_csv(pred_path)
# 常见列名兼容
rename_map = {
    "y_true": "y", "truth": "y", "target": "y",
    "prediction": "y_hat", "pred": "y_hat", "y_pred": "y_hat"
}
for k, v in rename_map.items():
    if k in df.columns and v not in df.columns:
        df = df.rename(columns={k: v})

if "y" not in df.columns or "y_hat" not in df.columns:
    raise ValueError("文件中未找到 y / y_hat 列，请检查。")

# 处理时间轴：有 date 就用，没有就用行号
time_axis = None
if "date" in df.columns:
    try:
        df["date"] = pd.to_datetime(df["date"])
        time_axis = df["date"]
    except Exception:
        pass
if time_axis is None:
    df["idx"] = np.arange(len(df))
    time_axis = df["idx"]

# 3) 计算残差与指标
df["residual"] = df["y_hat"] - df["y"]
mae  = mean_absolute_error(df["y"], df["y_hat"])
rmse = np.sqrt(((df["y_hat"] - df["y"])**2).mean())
r2   = r2_score(df["y"], df["y_hat"])

print(f"📊 指标 | MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

# 4) 画图并保存
# (1) 预测 vs 真实
plt.figure(figsize=(9,3))
plt.plot(time_axis, df["y"], label="True")
plt.plot(time_axis, df["y_hat"], label="Pred")
plt.legend(); plt.tight_layout()
plt.savefig(OUT_DIR/"pred_vs_true.png", dpi=300)

# (2) 残差随时间
plt.figure(figsize=(9,3))
plt.plot(time_axis, df["residual"])
plt.axhline(0, linestyle="--")
plt.tight_layout()
plt.savefig(OUT_DIR/"residual_time.png", dpi=300)

# (3) y_hat vs y 散点
plt.figure(figsize=(4,4))
plt.scatter(df["y"], df["y_hat"], s=10)
minv, maxv = df[["y","y_hat"]].min().min(), df[["y","y_hat"]].max().max()
plt.plot([minv,maxv],[minv,maxv], linestyle="--")
plt.xlabel("True"); plt.ylabel("Pred")
plt.tight_layout()
plt.savefig(OUT_DIR/"residual_scatter.png", dpi=300)

# (4) 残差直方图
plt.figure(figsize=(4,3))
plt.hist(df["residual"], bins=30)
plt.tight_layout()
plt.savefig(OUT_DIR/"residual_hist.png", dpi=300)

# 5) 指标保存
report = {"MAE":float(mae), "RMSE":float(rmse), "R2":float(r2), "source": pred_path}
(Path(OUT_DIR/"residual_report.json")
 .write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"))

print("🎉 残差图已保存：")
print(" - pred_vs_true.png")
print(" - residual_time.png")
print(" - residual_scatter.png")
print(" - residual_hist.png")
print(" - residual_report.json")

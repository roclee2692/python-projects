# -*- coding: utf-8 -*-
import json, joblib, warnings
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

IN_PQ   = r"D:\python-projects\q1_dataset.parquet"   # 若没有 parquet，就把上一版造特征的 DataFrame 直接用
OUT_DIR = Path(r"D:\python-projects"); OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_P = OUT_DIR / "lgbm_or_rf_model.pkl"
MET_P   = OUT_DIR / "final_metrics.json"
PRED_P  = OUT_DIR / "final_pred.csv"

warnings.filterwarnings("ignore")

# 1) 读数据
if IN_PQ.lower().endswith(".parquet"):
    data = pd.read_parquet(IN_PQ)
else:
    data = pd.read_csv(IN_PQ, parse_dates=["date"])  # 兜底
assert "sm5" in data.columns, "数据里找不到标签列 sm5"
feats = [c for c in data.columns if c not in ["sm5","date"]]

# 2) 时序 8:2 切分
data = data.sort_values(data.columns[0])  # 若第一列不是 date，也不会报错
split = int(len(data)*0.8)
Xtr, ytr = data.iloc[:split][feats], data.iloc[:split]["sm5"]
Xte, yte = data.iloc[split:][feats], data.iloc[split:]["sm5"]

# 3) 训练：优先 LightGBM，失败就 RF
model_name = ""
try:
    import lightgbm as lgb
    params = dict(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=127,
        min_data_in_leaf=100,
        feature_fraction=0.8,  # 只保留官方名；不同时给 colsample_bytree
        bagging_fraction=0.8,  # 不给 subsample
        bagging_freq=1,        # 不给 subsample_freq
        force_col_wise=True,   # 按日志建议
        random_state=42, n_jobs=-1,
        verbosity=-1           # 关闭冗余日志
    )
    model = lgb.LGBMRegressor(**params)
    # 新旧版本兼容：优先 callbacks 早停；不行再用 early_stopping_rounds
    try:
        model.fit(
            Xtr, ytr, eval_set=[(Xte, yte)], eval_metric="l1",
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
    except TypeError:
        model.fit(
            Xtr, ytr, eval_set=[(Xte, yte)], eval_metric="l1",
            early_stopping_rounds=100
        )
    model_name = "LightGBM"
except Exception as e:
    print("⚠️ LightGBM 不可用，退回随机森林：", e)
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1)
    model.fit(Xtr, ytr)
    model_name = "RandomForest"

# 4) 评估与保存
pred = model.predict(Xte)
mae  = mean_absolute_error(yte, pred)
rmse = float(np.sqrt(mean_squared_error(yte, pred)))
r2   = r2_score(yte, pred)

print(f"📊 最终指标 | MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  模型={model_name}")
joblib.dump(model, MODEL_P)
Path(MET_P).write_text(json.dumps({"MAE":mae,"RMSE":rmse,"R2":r2,"model":model_name}, ensure_ascii=False, indent=2), encoding="utf-8")
pd.DataFrame({"y":yte, "y_hat":pred}).to_csv(PRED_P, index=False, encoding="utf-8-sig")
print("✅ 已保存：", MODEL_P, MET_P, PRED_P)

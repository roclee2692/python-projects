import pandas as pd, numpy as np, json, joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

data = pd.read_parquet(r"D:\python-projects\q1_dataset.parquet")
feats = [c for c in data.columns if c != 'sm5']

tscv = TimeSeriesSplit(n_splits=3, test_size=int(len(data)*0.2), gap=0)
fold_metrics = []
for fold, (tr_idx, te_idx) in enumerate(tscv.split(data)):
    Xtr, ytr = data.iloc[tr_idx][feats], data.iloc[tr_idx]['sm5']
    Xte, yte = data.iloc[te_idx][feats], data.iloc[te_idx]['sm5']
    rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    fold_metrics.append({
        "fold": fold+1,
        "MAE":  mean_absolute_error(yte, pred),
        "RMSE": np.sqrt(((yte-pred)**2).mean()),
        "R2":   r2_score(yte, pred)
    })
print(pd.DataFrame(fold_metrics))
pd.DataFrame(fold_metrics).to_json(r"D:\python-projects\rf_metrics.json", orient="records", indent=2)
joblib.dump(rf, r"D:\python-projects\rf_model.pkl")

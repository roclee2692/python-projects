import joblib, shap, pandas as pd, matplotlib.pyplot as plt
model = joblib.load(r"D:\python-projects\lgbm_model.pkl")
data  = pd.read_parquet(r"D:\python-projects\q1_dataset.parquet")
feats = [c for c in data.columns if c != 'sm5']

# LightGBM 内置 FI 条形图
importances = model.feature_importances_
fi = pd.Series(importances, index=feats).sort_values(ascending=False)
fi.head(20).plot(kind='barh', figsize=(6,8))
plt.gca().invert_yaxis(); plt.tight_layout()
plt.savefig(r"D:\python-projects\feat_importance.png", dpi=300)

# SHAP (抽样 2000 行)
explainer = shap.TreeExplainer(model)
sample = data[feats].sample(n=min(2000, len(data)), random_state=1)
shap_values = explainer(sample)
shap.summary_plot(shap_values, sample, show=False)
plt.tight_layout()
plt.savefig(r"D:\python-projects\shap_summary.png", dpi=300)

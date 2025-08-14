import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as skm
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.impute import KNNImputer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def normalize_cell(x):
    if isinstance(x, str):
        y = (x.replace("\u00A0", " ").replace("\u3000", " ")
               .replace("\t", " ").replace("\r", " ").replace("\n", " ").strip())
        if y == "" or y.lower() in {"na", "n/a", "nan"}:
            return np.nan
        return y
    return x


CN_MAP = {
    '标识符': 'ID', '中间时间': 'N_Days', '患者状态': 'Status', '药物类型': 'Drug',
    '年龄（天）': 'Age', '存在腹水': 'Ascites', '肝肿大': 'Hepatomegaly', '存在': 'Spiders',
    '利尿剂与水肿': 'Edema', '血清胆红素': 'Bilirubin', '血清胆固醇': 'Cholesterol',
    '白蛋白': 'Albumin', '尿铜': 'Copper', '碱性磷酸酶': 'Alk_Phos', '甘油三酯': 'Triglycerides',
    '每立方血小板': 'Platelets', '凝血酶原时间': 'Prothrombin', '疾病组织学阶段': 'Stage',
}


def winsor(s, lo=0.01, hi=0.99):
    a, b = s.quantile(lo), s.quantile(hi)
    return s.clip(a, b)


def outlier_report(df, cols):
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) == 0:
            rows.append((c, 0, np.nan, np.nan, np.nan))
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        rows.append((c, int(((s < lo) | (s > hi)).sum()), lo, hi, float(s.skew())))
    return pd.DataFrame(rows, columns=["column", "outlier_count_3IQR", "low_bound", "high_bound", "skewness"])


def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--sheet", default="附录三cirrhosis")
    ap.add_argument("--header", type=int, default=1)
    ap.add_argument("--outdir", default="cirrhosis_run")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.outdir); ensure_dir(out_dir); ensure_dir(out_dir / "figs")

    # ---------- 读取 ----------
    ip = Path(args.input)
    if ip.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(ip, sheet_name=args.sheet, header=args.header)
        df = df.rename(columns={'Unnamed: 5': 'Sex', 'Unnamed: 15': 'SGOT'})
    else:
        df = pd.read_csv(ip)

    df = df.rename(columns={c: CN_MAP.get(c, c) for c in df.columns})
    df = df.rename(columns={'Tryglicerides': 'Triglycerides'})
    try:
        df = df.map(normalize_cell)
    except AttributeError:
        df = df.applymap(normalize_cell)

    num_cols = [c for c in ["N_Days", "Age", "Bilirubin", "Cholesterol", "Albumin",
                            "Copper", "Alk_Phos", "SGOT", "Triglycerides",
                            "Platelets", "Prothrombin", "Stage"] if c in df.columns]
    cat_cols = [c for c in ["Status", "Drug", "Sex", "Ascites",
                            "Hepatomegaly", "Spiders", "Edema"] if c in df.columns]

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---------- 缺失报告 ----------
    miss_counts = df[num_cols + cat_cols].isna().sum().sort_values(ascending=False)
    miss_rate = df[num_cols + cat_cols].isna().mean().reindex(miss_counts.index)
    pd.DataFrame({"column": miss_counts.index,
                  "missing_count": miss_counts.values,
                  "missing_rate": miss_rate.values}).to_csv(out_dir / "missing_report.csv",
                                                            index=False, encoding="utf-8-sig")

    # ---------- 中位数/众数 + 指示 ----------
    df_imp = df.copy()
    for c in num_cols:
        df_imp[c + "_na"] = df_imp[c].isna().astype(int)
        med = df_imp[c].median(skipna=True)
        df_imp[c] = df_imp[c].fillna(med)
    for c in cat_cols:
        df_imp[c] = df_imp[c].astype("string").fillna("Missing")
    df_imp["target"] = (df_imp["Stage"] >= 3).astype(int)
    df_imp.to_csv(out_dir / "imputed.csv", index=False, encoding="utf-8-sig")

    # ---------- 统计检验 ----------
    num_focus = [c for c in ['Bilirubin', 'Albumin', 'Prothrombin', 'Alk_Phos', 'SGOT'] if c in df_imp.columns]
    cat_focus = [c for c in ['Ascites', 'Edema'] if c in df_imp.columns]

    rows = []
    for c in num_focus:
        a = df_imp.loc[df_imp["target"] == 1, c].astype(float)
        b = df_imp.loc[df_imp["target"] == 0, c].astype(float)
        stat, p = mannwhitneyu(a, b, alternative='two-sided')
        rows.append([c, float(a.median()), float(b.median()),
                     float(a.median() - b.median()), float(stat), float(p),
                     int(a.shape[0]), int(b.shape[0])])
    pd.DataFrame(rows, columns=['feature', 'median(Stage≥3)', 'median(Stage<3)',
                                'median_diff', 'U_stat', 'p_value', 'n_pos', 'n_neg']
                 ).to_csv(out_dir / "u_tests.csv", index=False, encoding="utf-8-sig")

    rows = []
    for c in cat_focus:
        ct = pd.crosstab(df_imp["target"], df_imp[c]).fillna(0).astype(int)
        if ct.shape == (2, 2):
            chi2, p_chi, dof, exp = chi2_contingency(ct)
            if (exp < 5).any():
                _, p = fisher_exact(ct.values); method = "fisher_exact"
            else:
                method, p = "chi2", p_chi
        else:
            method, p = "chi2", chi2_contingency(ct)[1]
        rows.append([c, method, float(p)] + ct.values.flatten().tolist())
        ct.to_csv(out_dir / f"ct_{c}.csv", encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(out_dir / "cat_tests.csv", index=False, header=False, encoding="utf-8-sig")

    # ---------- 作图 ----------
    def save_hist_and_box(col):
        a = df_imp[df_imp["target"] == 1][col].astype(float)
        b = df_imp[df_imp["target"] == 0][col].astype(float)
        plt.figure()
        bins = np.histogram(np.hstack([a.values, b.values]), bins=30)[1]
        plt.hist(a, bins=bins, alpha=0.5, label='Stage≥3')
        plt.hist(b, bins=bins, alpha=0.5, label='Stage<3')
        plt.title(f"{col} - Histogram by target"); plt.xlabel(col); plt.ylabel("Count"); plt.legend()
        plt.savefig(out_dir / "figs" / f"{col}_hist.png", dpi=160, bbox_inches="tight"); plt.close()
        plt.figure()
        plt.boxplot([a, b], tick_labels=['Stage≥3', 'Stage<3'], showmeans=True)
        plt.title(f"{col} - Boxplot by target"); plt.ylabel(col)
        plt.savefig(out_dir / "figs" / f"{col}_box.png", dpi=160, bbox_inches="tight"); plt.close()

    for c in num_focus:
        save_hist_and_box(c)

    for c in cat_focus:
        ct = pd.crosstab(df_imp["target"], df_imp[c]).reindex(index=[1, 0]).fillna(0)
        props = (ct.T / ct.T.sum()).T
        ax = props.plot(kind='bar', stacked=True, legend=True)
        plt.title(f"{c} - Proportions by target"); plt.xlabel("target"); plt.ylabel("Proportion")
        plt.tight_layout()
        plt.savefig(out_dir / "figs" / f"{c}_proportion.png", dpi=160, bbox_inches="tight"); plt.close()

    # ---------- 建模 ----------
    exclude = {'ID', 'Stage', 'Stage_na', 'N_Days', 'N_Days_na'}
    all_num = df_imp.select_dtypes(include=[np.number]).columns.tolist()
    num_feats = [c for c in all_num if c not in exclude and c != "target"]
    cat_feats = [c for c in df_imp.columns if c not in all_num + ["target"]]

    X = df_imp[num_feats + cat_feats].copy()
    y = df_imp["target"].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=args.seed)

    oh = OneHotEncoder(handle_unknown='ignore')
    scaler = StandardScaler()

    heavy = [c for c in ["Bilirubin", "Alk_Phos", "Copper", "SGOT", "Triglycerides"] if c in X.columns]

    pre = ColumnTransformer([('num', scaler, num_feats), ('cat', oh, cat_feats)])
    logit_base = Pipeline([('pre', pre),
                           ('clf', LogisticRegression(max_iter=400, class_weight='balanced'))])

    X_train_log, X_test_log = X_train.copy(), X_test.copy()
    for c in heavy:
        X_train_log[c] = np.log1p(np.clip(X_train_log[c].astype(float), a_min=0, a_max=None))
        X_test_log[c] = np.log1p(np.clip(X_test_log[c].astype(float), a_min=0, a_max=None))
    logit_log = Pipeline([('pre', pre),
                          ('clf', LogisticRegression(max_iter=400, class_weight='balanced'))])

    pre_tree = ColumnTransformer([('cat', oh, cat_feats)], remainder='passthrough')
    try:
        import xgboost as xgb
        tree = Pipeline([
            ('pre', pre_tree),
            ('clf', xgb.XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                objective='binary:logistic', eval_metric='logloss',
                random_state=args.seed, n_jobs=4,
                scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum())
            ))
        ])
        tree_name = "XGBoost"
    except Exception:
        from sklearn.ensemble import RandomForestClassifier
        tree = Pipeline([
            ('pre', pre_tree),
            ('clf', RandomForestClassifier(
                n_estimators=500, random_state=args.seed, class_weight='balanced'
            ))
        ])
        tree_name = "RandomForest"

    logit_base.fit(X_train, y_train)
    logit_log.fit(X_train_log, y_train)
    tree.fit(X_train, y_train)

    # ---------- 评估函数 ----------
    def eval_model(model, X_tr, X_te, y_tr, y_te, name, out_prefix):
        proba_tr = model.predict_proba(X_tr)[:, 1]
        proba_te = model.predict_proba(X_te)[:, 1]

        auc_tr = skm.roc_auc_score(y_tr, proba_tr)
        auc_te = skm.roc_auc_score(y_te, proba_te)
        pr_tr = skm.average_precision_score(y_tr, proba_tr)
        pr_te = skm.average_precision_score(y_te, proba_te)
        brier = skm.brier_score_loss(y_te, proba_te)

        fpr, tpr, _ = skm.roc_curve(y_te, proba_te)
        plt.figure(); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], '--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'{name} ROC (test)')
        plt.savefig(out_dir / "figs" / f"{out_prefix}_roc.png", dpi=160, bbox_inches="tight"); plt.close()

        prec, rec, _ = skm.precision_recall_curve(y_te, proba_te)
        plt.figure(); plt.plot(rec, prec)
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'{name} PR (test)')
        plt.savefig(out_dir / "figs" / f"{out_prefix}_pr.png", dpi=160, bbox_inches="tight"); plt.close()

        frac_pos, mean_pred = calibration_curve(y_te, proba_te, n_bins=10, strategy='quantile')
        plt.figure(); plt.plot(mean_pred, frac_pos, marker='o'); plt.plot([0, 1], [0, 1], '--')
        plt.xlabel('Predicted'); plt.ylabel('Observed'); plt.title(f'{name} Calibration (test)')
        plt.savefig(out_dir / "figs" / f"{out_prefix}_cal.png", dpi=160, bbox_inches="tight"); plt.close()

        return {"model": name, "AUC_train": auc_tr, "AUC_test": auc_te,
                "PR_AUC_train": pr_tr, "PR_AUC_test": pr_te, "Brier_test": brier}

    metrics = []
    metrics.append(eval_model(logit_base, X_train, X_test, y_train, y_test,
                              "Logistic (z-score)", "logit_base"))
    metrics.append(eval_model(logit_log, X_train_log, X_test_log, y_train, y_test,
                              "Logistic (log heavy + z-score)", "logit_log"))
    metrics.append(eval_model(tree, X_train, X_test, y_train, y_test,
                              tree_name, "tree_model"))
    pd.DataFrame(metrics).round(4).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    # ---------- Logistic 系数（OR） ----------
    pipe_l = logit_base
    clf_l = pipe_l.named_steps['clf']
    pre_l = pipe_l.named_steps['pre']
    num_names = list(pre_l.transformers_[0][2])
    oh_l = pre_l.transformers_[1][1]
    cat_names = list(oh_l.get_feature_names_out(pre_l.transformers_[1][2]))
    feat_names = num_names + cat_names
    coef = pd.DataFrame({"feature": feat_names, "coef": clf_l.coef_.ravel()})
    coef["OR"] = np.exp(coef["coef"])
    coef.sort_values("OR", ascending=False).to_csv(out_dir / "logistic_coefficients.csv",
                                                   index=False, encoding="utf-8-sig")

    # ---------- Permutation 重要性（可选） ----------
    try:
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(tree, X_test, y_test, n_repeats=5,
                                      random_state=args.seed, n_jobs=2)
        imp = pd.DataFrame({"feature": list(X_test.columns),
                            "importance": perm.importances_mean})
        imp.sort_values("importance", ascending=False).to_csv(out_dir / "perm_importance_tree.csv",
                                                              index=False, encoding="utf-8-sig")
    except Exception:
        pass

    # ---------- 三档阈值表 ----------
    def thresholds_table(model, Xte, yte, out_file):
        proba = model.predict_proba(Xte)[:, 1]
        prec, rec, thr = skm.precision_recall_curve(yte, proba)
        fpr, tpr, thr_roc = skm.roc_curve(yte, proba)

        def pick_recall(target):
            i = np.argmin(np.abs(rec - target))
            return thr[i - 1] if i > 0 else thr[0]

        th_high = pick_recall(0.90)
        j = np.argmax(tpr - fpr); th_bal = thr_roc[j]
        j2 = np.argmin(np.abs(fpr - 0.05)); th_low = thr_roc[j2]

        rows = []
        for label, th in [("high_recall", th_high), ("balanced", th_bal), ("low_fpr", th_low)]:
            yhat = (proba >= th).astype(int)
            TP = int(((yhat == 1) & (yte == 1)).sum())
            FP = int(((yhat == 1) & (yte == 0)).sum())
            TN = int(((yhat == 0) & (yte == 0)).sum())
            FN = int(((yhat == 0) & (yte == 1)).sum())
            P = TP / (TP + FP + 1e-9); R = TP / (TP + FN + 1e-9)
            rows.append([label, th, TP, FP, TN, FN, P, R])

        pd.DataFrame(rows, columns=["setting", "threshold", "TP", "FP", "TN", "FN",
                                    "precision", "recall"]
                     ).to_csv(out_dir / out_file, index=False, encoding="utf-8-sig")

    thresholds_table(tree, X_test, y_test, "thresholds_tree.csv")
    thresholds_table(logit_base, X_test, y_test, "thresholds_logit.csv")

    # ---------- 异常值报告 ----------
    outliers = outlier_report(df_imp, [c for c in num_cols if c != "Stage"])
    outliers.to_csv(out_dir / "outliers_3IQR.csv", index=False, encoding="utf-8-sig")

    # ---------- 4.2 灵敏度 & 4.3 改进 ----------
    def _prep_xy(df_any):
        exclude_l = {'ID', 'Stage', 'Stage_na', 'N_Days', 'N_Days_na'}
        all_num_l = df_any.select_dtypes(include=[np.number]).columns.tolist()
        num_feats_l = [c for c in all_num_l if c not in exclude_l and c != "target"]
        cat_feats_l = [c for c in df_any.columns if c not in all_num_l + ["target"]]
        X_l = df_any[num_feats_l + cat_feats_l].copy()
        y_l = df_any["target"].copy()
        return X_l, y_l, num_feats_l, cat_feats_l

    def _fit_eval_logit_and_tree(Xtr, Xte, ytr, yte, numf, catf, tag):
        oh_ = OneHotEncoder(handle_unknown='ignore'); scaler_ = StandardScaler()
        pre_ = ColumnTransformer([('num', scaler_, numf), ('cat', oh_, catf)])
        logit_ = Pipeline([('pre', pre_), ('clf', LogisticRegression(max_iter=400, class_weight='balanced'))])

        pre_tree_ = ColumnTransformer([('cat', oh_, catf)], remainder='passthrough')
        try:
            import xgboost as xgb
            tree_ = Pipeline([('pre', pre_tree_),
                              ('clf', xgb.XGBClassifier(
                                  n_estimators=400, max_depth=4, learning_rate=0.05,
                                  subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                                  objective='binary:logistic', eval_metric='logloss',
                                  random_state=args.seed, n_jobs=4,
                                  scale_pos_weight=float((ytr == 0).sum() / (ytr == 1).sum())
                              ))])
            name_t = "XGBoost"
        except Exception:
            from sklearn.ensemble import RandomForestClassifier
            tree_ = Pipeline([('pre', pre_tree_),
                              ('clf', RandomForestClassifier(
                                  n_estimators=500, random_state=args.seed, class_weight='balanced'))])
            name_t = "RandomForest"

        logit_.fit(Xtr, ytr); tree_.fit(Xtr, ytr)
        def _m(model, Xte_, yte_, name_):
            p = model.predict_proba(Xte_)[:, 1]
            return {"exp": tag, "model": name_,
                    "AUC_test": skm.roc_auc_score(yte_, p),
                    "PR_AUC_test": skm.average_precision_score(yte_, p),
                    "Brier_test": skm.brier_score_loss(yte_, p)}
        return _m(logit_, Xte, yte, "Logistic"), _m(tree_, Xte, yte, name_t)

    base_metrics = pd.read_csv(out_dir / "metrics.csv")
    base_tree_auc = float(base_metrics.loc[base_metrics.model.str.contains("XGBoost|RandomForest"), 'AUC_test'].iloc[0])
    base_logit_auc = float(base_metrics.loc[base_metrics.model.str.contains("Logistic \\(z-score\\)"), 'AUC_test'].iloc[0])

    sens_rows = []

    # 缺失策略：KNN
    df_knn = df.copy()
    num_for_knn = [c for c in num_cols if c in df_knn.columns]
    arr = KNNImputer(n_neighbors=5).fit_transform(df_knn[num_for_knn])
    df_knn_imp = df_knn.copy()
    for i, c in enumerate(num_for_knn): df_knn_imp[c] = arr[:, i]
    for c in cat_cols: df_knn_imp[c] = df_knn_imp[c].astype("string").fillna("Missing")
    df_knn_imp["target"] = (pd.to_numeric(df_knn_imp["Stage"], errors="coerce") >= 3).astype(int)
    Xs, ys, nf, cf = _prep_xy(df_knn_imp)
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.2, stratify=ys, random_state=args.seed)
    m1, m2 = _fit_eval_logit_and_tree(Xtr, Xte, ytr, yte, nf, cf, "impute_KNN"); sens_rows += [m1, m2]

    # 去掉 Copper
    df_drop = df_imp.copy()
    for c in ["Copper", "Copper_na"]:
        if c in df_drop.columns: df_drop = df_drop.drop(columns=c)
    Xs, ys, nf, cf = _prep_xy(df_drop)
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.2, stratify=ys, random_state=args.seed)
    m1, m2 = _fit_eval_logit_and_tree(Xtr, Xte, ytr, yte, nf, cf, "drop_Copper"); sens_rows += [m1, m2]

    # 加派生特征
    df_feat = df_imp.copy()
    df_feat["Bili_Alb"] = df_feat["Bilirubin"] / df_feat["Albumin"].replace(0, np.nan)
    df_feat["Alk_SGOT"] = df_feat["Alk_Phos"] / df_feat["SGOT"].replace(0, np.nan)
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    for c in ["Bili_Alb", "Alk_SGOT"]:
        df_feat[c] = df_feat[c].fillna(df_feat[c].median())
    Xs, ys, nf, cf = _prep_xy(df_feat)
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.2, stratify=ys, random_state=args.seed)
    m1, m2 = _fit_eval_logit_and_tree(Xtr, Xte, ytr, yte, nf, cf, "add_derived"); sens_rows += [m1, m2]

    # log1p 影响（日志中已有） —— 全段替换为以下 5 行
    logit_auc_log = base_metrics.loc[base_metrics.model.str.contains(r"Logistic \(log heavy"), "AUC_test"].iloc[0]
    pr_log = base_metrics.loc[base_metrics.model.str.contains(r"Logistic \(log heavy"), "PR_AUC_test"].iloc[0]
    brier_log = base_metrics.loc[base_metrics.model.str.contains(r"Logistic \(log heavy"), "Brier_test"].iloc[0]
    sens_rows.append({"exp": "log1p_heavy(Logistic)", "model": "Logistic_log_vs_base",
                      "AUC_test": logit_auc_log, "PR_AUC_test": pr_log, "Brier_test": brier_log})

    sens_df = pd.DataFrame(sens_rows)
    def _delta(row):
        base = base_tree_auc if ("XGBoost" in row["model"] or "RandomForest" in row["model"]) else base_logit_auc
        return row["AUC_test"] - base
    sens_df["delta_AUC_vs_base"] = sens_df.apply(_delta, axis=1)
    sens_df.to_csv(out_dir / "sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    # ---- 改进汇总：至少写入2项（派生特征、基线树校准），即使没有 xgboost 也有结果 ----
    imp_rows = []

    # 1) 派生特征带来的提升（用 sensitivity 里已有的 add_derived 结果）
    try:
        tree_add_auc = float(
            sens_df[(sens_df.exp == "add_derived") & (~sens_df.model.str.contains("Logistic"))]["AUC_test"].iloc[0]
        )
        imp_rows.append({
            "improve": "add_derived(Bili_Alb,Alk_SGOT)",
            "AUC_test": tree_add_auc,
            "delta_AUC": tree_add_auc - base_tree_auc
        })
    except Exception:
        pass

    # 2) 对“当前 tree 管道”（可能是 XGBoost 或 RandomForest）做概率校准（isotonic）
    Xs, ys, nf, cf = _prep_xy(df_imp)
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.2, stratify=ys, random_state=args.seed)
    cal = CalibratedClassifierCV(tree, method="isotonic", cv=3)  # 直接包在已建好的 tree 上
    cal.fit(Xtr, ytr)
    p_cal = cal.predict_proba(Xte)[:, 1]
    imp_rows.append({
        "improve": "calibration_isotonic_on_baseline_tree",
        "AUC_test": skm.roc_auc_score(yte, p_cal),
        "delta_AUC": skm.roc_auc_score(yte, p_cal) - base_tree_auc,
        "Brier_test": skm.brier_score_loss(yte, p_cal)
    })

    pd.DataFrame(imp_rows).to_csv(out_dir / "improvement_summary.csv", index=False, encoding="utf-8-sig")

    # ---------- 改进：早停和校准（整块替换） ----------
    imp_rows = []

    # 1) 派生特征带来的提升（来自 sensitivity 的 add_derived 行）
    try:
        tree_add_auc = \
        sens_df[(sens_df.exp == "add_derived") & (~sens_df.model.str.contains("Logistic"))]["AUC_test"].iloc[0]
        imp_rows.append({
            "improve": "add_derived(Bili_Alb,Alk_SGOT)",
            "AUC_test": float(tree_add_auc),
            "delta_AUC": float(tree_add_auc) - float(base_tree_auc)
        })
    except Exception as e:
        print("add_derived line missing:", e)

    # 2) 直接对“当前树管道”（XGB 或 RF）做概率校准（isotonic）——必定落地一行
    cal = CalibratedClassifierCV(tree, method="isotonic", cv=3)  # 用已训练好的 tree 管道
    cal.fit(X_train, y_train)
    p_cal = cal.predict_proba(X_test)[:, 1]
    imp_rows.append({
        "improve": "calibration_isotonic_on_baseline_tree",
        "AUC_test": skm.roc_auc_score(y_test, p_cal),
        "delta_AUC": skm.roc_auc_score(y_test, p_cal) - float(base_tree_auc),
        "Brier_test": skm.brier_score_loss(y_test, p_cal)
    })

    # 3) 若可用 XGBoost：调参（无早停，保证兼容）+ 双校准
    try:
        import xgboost as xgb
        oh2 = OneHotEncoder(handle_unknown='ignore')
        pre2 = ColumnTransformer([('cat', oh2, cat_feats)], remainder='passthrough')

        Xtr_p = pre2.fit_transform(X_train);
        Xte_p = pre2.transform(X_test)
        clf = xgb.XGBClassifier(
            n_estimators=800,  # 轻量调参
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=args.seed,
            scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()),
            n_jobs=4
        )
        # 直接拟合（不传 callbacks / early_stopping_rounds，兼容 3.x）
        clf.fit(Xtr_p, y_train, eval_set=[(Xte_p, y_test)], verbose=False)

        p_xgb = clf.predict_proba(Xte_p)[:, 1]
        auc_tuned = skm.roc_auc_score(y_test, p_xgb);
        brier_tuned = skm.brier_score_loss(y_test, p_xgb)
        imp_rows.append({
            "improve": "xgb_tuned",
            "AUC_test": auc_tuned,
            "delta_AUC": auc_tuned - float(base_tree_auc),
            "Brier_test": brier_tuned
        })

        # 对 XGB 再做双校准
        for method in ["isotonic", "sigmoid"]:
            calx = CalibratedClassifierCV(clf, method=method, cv=3)
            calx.fit(Xtr_p, y_train)
            p2 = calx.predict_proba(Xte_p)[:, 1]
            imp_rows.append({
                "improve": f"calibration_{method}_on_xgb",
                "AUC_test": skm.roc_auc_score(y_test, p2),
                "delta_AUC": skm.roc_auc_score(y_test, p2) - float(base_tree_auc),
                "Brier_test": skm.brier_score_loss(y_test, p2)
            })
    except Exception as e:
        print("xgboost improve block skipped:", e)

    pd.DataFrame(imp_rows).to_csv(out_dir / "improvement_summary.csv", index=False, encoding="utf-8-sig")
    print(f"Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()

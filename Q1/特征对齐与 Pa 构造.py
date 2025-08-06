# -*- coding: utf-8 -*-
"""
读取题干“表2”（某一天逐小时气象），自动：
- 识别第二行表头（有大标题时）
- 统一列名、数值化、时间(h)->datetime
- 计算 VPD；DD->角度->sin/cos；Pa 缺则用 P/Po 的3小时差构造(前2小时置0并打标)
- 加载 D:\python-projects\model.pkl 与 feature_list.json
- 特征对齐并预测，导出 带预测 的表格
输出目录：D:\python-projects\
"""

import re, json, joblib, numpy as np, pandas as pd
from pathlib import Path

# ========= 路径配置 =========
IN_TABLE2   = r"C:\Users\Raelon\OneDrive\文档\表2.xlsx"  # 题干给的“表2”文件（可为 .xlsx 或 .csv）
SHEET_NAME  = 0                                # Excel 的表单序号/名称
DAY         = "2021-07-15"                     # 表2对应日期（务必正确）
OUT_DIR     = Path(r"D:\python-projects")      # 统一输出目录
MODEL_P     = OUT_DIR / "model.pkl"
FEATS_P     = OUT_DIR / "feature_list.json"

OUT_FEAT_CSV= OUT_DIR / "table2_features.csv"
OUT_PRED_CSV= OUT_DIR / "table2_pred.csv"
OUT_PRED_XLSX= OUT_DIR / "表2_带预测.xlsx"
REPORT_JSON = OUT_DIR / "table2_features_report.json"

# ========= 小工具 =========
CARDINALS = {"北":0.0,"东北":45.0,"东":90.0,"东南":135.0,"南":180.0,"西南":225.0,"西":270.0,"西北":315.0}

def read_with_header_fix(path, sheet):
    """首行若是大标题（以#或‘气象站’开头），则用第二行为表头(header=1)。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到表2文件：{path}")
    if path.lower().endswith((".xlsx",".xls")):
        probe = pd.read_excel(path, sheet_name=sheet, header=0, nrows=1)
        first_col = str(probe.columns[0])
        header_row = 1 if first_col.startswith("#") or "气象站" in first_col else 0
        df = pd.read_excel(path, sheet_name=sheet, header=header_row)
    else:
        df = pd.read_csv(path)
    return df

def unify_columns(df):
    """统一常见列名到题干字段"""
    rename = {
        "时间（h）":"hour","时间(h)":"hour","时间":"hour","hour":"hour",
        "T":"T","Po":"Po","P":"P","Pa":"Pa","U":"U","DD":"DD","Ff":"Ff",
        "RRR":"RRR","RR":"RRR","R":"RRR","Td":"Td","VV":"VV"
    }
    m = {k:v for k,v in rename.items() if k in df.columns}
    return df.rename(columns=m)

def parse_dd_to_deg(s):
    if pd.isna(s): return np.nan
    s = str(s).strip()
    # 先匹配“从……方向吹来的风”
    m = re.search(r"从(.+?)方向?吹来的风", s)
    core = m.group(1) if m else s
    # 去掉“轻风/微风/风”字样
    core = re.sub(r"(轻风|微风|和风|清劲风|强风|疾风|风)", "", core)
    hit = None
    for k in sorted(CARDINALS.keys(), key=len, reverse=True):
        if k in core:
            hit = k; break
    if hit is None:
        # 兜底：找任意一个方位字
        for k in CARDINALS:
            if k in core:
                hit = k; break
    return CARDINALS.get(hit, np.nan)

def compute_vpd_kpa(T, U):
    es = 0.6108 * np.exp(17.27*T/(T+237.3))
    ea = es * (U/100.0)
    return es - ea

# ========= 主流程 =========
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("🚀 读取表2...")
    raw = read_with_header_fix(IN_TABLE2, SHEET_NAME)
    print(f"✅ 原始表2形状：{raw.shape}")
    print("原始列名预览：", list(raw.columns)[:12])

    tab = unify_columns(raw)

    # hour -> datetime
    if "hour" not in tab.columns:
        raise ValueError("❌ 表2缺少‘时间(h)’列，请确认列名或手动改成‘时间(h)’/‘hour’。")
    tab["hour"] = tab["hour"].astype(str).str.extract(r"(\d{1,2})").iloc[:,0].astype(int).clip(0,23)
    tab["hour_str"] = tab["hour"].astype(str).str.zfill(2)
    tab["time"] = pd.to_datetime(tab["hour_str"].radd(f"{DAY} ")+":00")

    # 数值化
    for c in ["T","U","Ff","Po","P","Pa","RRR","Td","VV"]:
        if c in tab.columns:
            tab[c] = pd.to_numeric(tab[c], errors="coerce")

    # VPD
    if not {"T","U"}.issubset(tab.columns):
        raise ValueError("❌ 表2至少需要 T 与 U 才能计算 VPD。")
    tab["VPD"] = compute_vpd_kpa(tab["T"], tab["U"])

    # 风向
    if "DD" in tab.columns:
        tab["DD_deg"] = tab["DD"].apply(parse_dd_to_deg)
        tab["sin_dir"] = np.sin(np.radians(tab["DD_deg"]))
        tab["cos_dir"] = np.cos(np.radians(tab["DD_deg"]))
    else:
        tab["DD_deg"] = np.nan
        tab["sin_dir"] = np.nan
        tab["cos_dir"] = np.nan

    # Pa：若缺则用 P 或 Po 的3小时差构造（前2小时置0）
    if ("Pa" not in tab.columns) or tab["Pa"].isna().all():
        src = "P" if "P" in tab.columns else ("Po" if "Po" in tab.columns else None)
        if src:
            tab = tab.sort_values("time")
            tab["Pa"] = tab[src] - tab[src].shift(3)
            tab.loc[tab["hour"] < 3, "Pa"] = 0.0
            tab["has_Pa"] = 0
            print(f"ℹ️ 未提供 Pa，用 {src} 构造 3 小时差；前 2 小时置 0。")
        else:
            tab["Pa"] = 0.0
            tab["has_Pa"] = 0
            print("ℹ️ 无 P/Po 可构造 Pa，统一置 0。")
    else:
        tab["has_Pa"] = tab["Pa"].notna().astype(int)
        tab["Pa"] = tab["Pa"].fillna(0.0)

    # RRR 缺失置0
    if "RRR" not in tab.columns:
        tab["RRR"] = 0.0
    else:
        tab["RRR"] = tab["RRR"].fillna(0.0)

    # 导出特征中间表
    used_cols = ["time","hour","T","U","Ff","Po","P","Pa","has_Pa","RRR","DD","DD_deg","sin_dir","cos_dir","VPD","Td","VV"]
    used_cols = [c for c in used_cols if c in tab.columns]
    feat = tab[used_cols].copy()
    feat.to_csv(OUT_FEAT_CSV, index=False, encoding="utf-8-sig")

    # 读取模型与特征清单
    model = joblib.load(MODEL_P)
    feats = json.loads(Path(FEATS_P).read_text(encoding="utf-8"))["feats"]

    # 将“当前小时”的基础列映射到训练用的 *_lag1 列；其余（累计/差分）用 0 兜底
    X = pd.DataFrame(index=feat.index)
    base_map = {
        "T":"T_lag1","U":"U_lag1","Ff":"Ff_lag1","VV":"VV_lag1","Td":"Td_lag1","VPD":"VPD_lag1",
        "RRR":"RRR_lag1","Po":"Po_lag1","P":"P_lag1","Pa":"Pa_lag1","T_max":"T_max_lag1","T_min":"T_min_lag1"
    }
    for base, lagname in base_map.items():
        if (lagname in feats) and (base in feat.columns):
            X[lagname] = feat[base]
    # 其余训练特征补 0
    missing_for_pred = []
    for col in feats:
        if col not in X.columns:
            X[col] = 0.0
            missing_for_pred.append(col)
    X = X[feats]

    # 预测
    pred = model.predict(X)

    # 回写到原表（保留原样+新增列）
    out = raw.copy()
    out["predicted_5cm_SM"] = pred
    out.to_csv(OUT_PRED_CSV, index=False, encoding="utf-8-sig")
    try:
        with pd.ExcelWriter(OUT_PRED_XLSX, engine="openpyxl") as w:
            out.to_excel(w, index=False, sheet_name="表2_带预测")
    except Exception:
        pass

    # 报告
    report = {
        "rows": int(len(feat)),
        "dd_mapped": int(feat["sin_dir"].notna().sum()),
        "pa_nonzero": int((feat["Pa"].abs() > 1e-9).sum()),
        "vpd_min": float(np.nanmin(feat["VPD"])) if "VPD" in feat.columns else None,
        "vpd_mean": float(np.nanmean(feat["VPD"])) if "VPD" in feat.columns else None,
        "vpd_max": float(np.nanmax(feat["VPD"])) if "VPD" in feat.columns else None,
        "missing_features_filled0": missing_for_pred
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 控制台摘要
    print("\n🎉 处理与预测完成")
    print(f"- 特征中间表：{OUT_FEAT_CSV}")
    print(f"- 带预测CSV：{OUT_PRED_CSV}")
    print(f"- 带预测Excel：{OUT_PRED_XLSX}")
    print(f"- 报告：{REPORT_JSON}")
    print(f"- 风向成功映射：{report['dd_mapped']} / {report['rows']}")
    print(f"- Pa 非零小时数：{report['pa_nonzero']}")
    print(f"- VPD范围：min={report['vpd_min']:.3f} mean={report['vpd_mean']:.3f} max={report['vpd_max']:.3f}")
    if report["missing_features_filled0"]:
        print("⚠️ 以下训练特征在表2中无法获得，已置0：")
        print(report["missing_features_filled0"][:20], "…（如有更多略）")

if __name__ == "__main__":
    main()

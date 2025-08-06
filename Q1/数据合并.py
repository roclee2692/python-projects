# -*- coding: utf-8 -*-
"""
B. 将逐时气象聚合到“日”并与表单二合并
输入：
  - q1_clean.csv            （逐时气象清洗结果，含列：time,T,U,Ff,ff10,VV,Td,RRR,Po,P,Pa,...）
  - soil_daily_clean.csv    （表单二清洗结果，含列：date, sm5,..., st40, obs48 等）
输出：
  - met_daily_features.csv  （聚合后的日尺度气象特征）
  - q1_daily_merged.csv     （与土壤日均合并后的建模数据）
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import json

# ======== 可配置路径 ========
MET_HOURLY_CSV   = r"D:\python-projects\q1_clean.csv"
SOIL_DAILY_CSV   = r"D:\python-projects\soil_daily_clean.csv"
OUT_MET_DAILY    = "met_daily_features.csv"
OUT_MERGED_DAILY = "q1_daily_merged.csv"
REPORT_JSON      = "q1_daily_merge_report.json"

def fail(msg, code=1):
    print("❌", msg)
    sys.exit(code)

def ensure_cols(df, need, name):
    miss = [c for c in need if c not in df.columns]
    if miss:
        print(f"⚠️  警告：{name} 缺少列：{miss}（将尽量继续）")
    return miss

def compute_vpd_kpa(T, Td=None, U=None):
    """返回 VPD(kPa)。优先用 Td，没有则用 U 近似。缺失则返回 NaN。"""
    es = 0.6108 * np.exp(17.27*T/(T+237.3))
    if Td is not None:
        ea = 0.6108 * np.exp(17.27*Td/(Td+237.3))
    elif U is not None:
        ea = es * (U/100.0)
    else:
        return np.nan
    return es - ea

def main():
    # 1) 读逐时气象
    p1 = Path(MET_HOURLY_CSV)
    if not p1.exists():
        fail(f"找不到逐时气象文件：{MET_HOURLY_CSV}")
    met_h = pd.read_csv(MET_HOURLY_CSV)
    if "time" not in met_h.columns:
        fail("逐时气象缺少 time 列，请确认你已执行步骤1生成的 q1_clean.csv。")

    # 解析时间索引
    met_h["time"] = pd.to_datetime(met_h["time"], errors="coerce")
    met_h = met_h.dropna(subset=["time"]).sort_values("time").set_index("time")
    # 若带时区，去掉时区以便 resample
    if met_h.index.tz is not None:
        met_h.index = met_h.index.tz_convert(None)

    # 2) 关键列存在性检查
    need_any_for_vpd = (("T" in met_h.columns) and (("Td" in met_h.columns) or ("U" in met_h.columns)))
    need_basic = ["T","U","Ff","ff10","VV","RRR","Po","P","Pa"]
    ensure_cols(met_h, need_basic, "逐时气象")

    # 3) 若无 VPD 列则按小时先算一个
    if "VPD" not in met_h.columns and need_any_for_vpd:
        T = pd.to_numeric(met_h.get("T"), errors="coerce")
        Td = pd.to_numeric(met_h.get("Td"), errors="coerce") if "Td" in met_h.columns else None
        U  = pd.to_numeric(met_h.get("U"),  errors="coerce") if "U"  in met_h.columns else None
        met_h["VPD"] = compute_vpd_kpa(T, Td=Td, U=U)

    # 4) 日尺度聚合（均值/极值/总量）
    agg_map = {}
    if "T"   in met_h: agg_map["T"]   = "mean"
    if "U"   in met_h: agg_map["U"]   = "mean"
    if "Ff"  in met_h: agg_map["Ff"]  = "mean"
    if "ff10"in met_h: agg_map["ff10"]= "max"
    if "VV"  in met_h: agg_map["VV"]  = "mean"
    if "Td"  in met_h: agg_map["Td"]  = "mean"
    if "VPD" in met_h: agg_map["VPD"] = "mean"
    if "RRR" in met_h: agg_map["RRR"] = "sum"   # 日降水量
    if "Po"  in met_h: agg_map["Po"]  = "mean"
    if "P"   in met_h: agg_map["P"]   = "mean"
    if "Pa"  in met_h: agg_map["Pa"]  = "mean"

    met_d = met_h.resample("D").agg(agg_map)

    # 附加：日最高/最低气温
    if "T" in met_h:
        met_d["T_max"] = met_h["T"].resample("D").max()
        met_d["T_min"] = met_h["T"].resample("D").min()

    met_d = met_d.reset_index().rename(columns={"time":"date"})
    met_d["date"] = pd.to_datetime(met_d["date"]).dt.date
    met_d.to_csv(OUT_MET_DAILY, index=False, encoding="utf-8-sig")
    print(f"✅ 已输出日尺度气象特征：{OUT_MET_DAILY}（{len(met_d)} 天）")

    # 5) 读取表单二（日均土壤）
    p2 = Path(SOIL_DAILY_CSV)
    if not p2.exists():
        fail(f"找不到土壤日均文件：{SOIL_DAILY_CSV}（请先运行表单二清洗脚本生成）")
    soil = pd.read_csv(SOIL_DAILY_CSV)
    if "date" not in soil.columns:
        # 兼容直接从 Excel 导出的情况
        if "DATE" in soil.columns:
            soil = soil.rename(columns={"DATE":"date"})
        else:
            fail("土壤日均数据缺少 date 列。")
    soil["date"] = pd.to_datetime(soil["date"]).dt.date

    # 6) 合并（inner，确保两边都有）
    merged = soil.merge(met_d, on="date", how="inner")
    merged.to_csv(OUT_MERGED_DAILY, index=False, encoding="utf-8-sig")
    print(f"🎉 已输出合并数据：{OUT_MERGED_DAILY}（{len(merged)} 天，含土壤与日气象特征）")

    # 7) 简短报告
    rpt = {
        "met_daily_rows": int(len(met_d)),
        "soil_daily_rows": int(len(soil)),
        "merged_rows": int(len(merged)),
        "date_range_met": [str(met_d["date"].min()), str(met_d["date"].max())],
        "date_range_soil": [str(soil["date"].min()), str(soil["date"].max())],
        "date_range_merged": [str(merged["date"].min()), str(merged["date"].max())],
        "columns_merged": list(merged.columns)
    }
    Path(REPORT_JSON).write_text(json.dumps(rpt, ensure_ascii=False, indent=2), encoding="utf-8")
    print("📄 合并报告：", REPORT_JSON)

    # 8) 贴心提示：避免“未来信息泄漏”的做法
    print("\n🔒 提示：如果要用当日气象预测当日 sm5，严格做法是使用“前一日/前几日”的聚合特征，")
    print("   例如将 met_d 全部列在训练前 shift(1) 成为昨天特征，再与当日 sm5 合并。")
    print("   示例：merged[[col for col in met_d_cols]] = merged[met_d_cols].shift(1)")

if __name__ == "__main__":
    main()

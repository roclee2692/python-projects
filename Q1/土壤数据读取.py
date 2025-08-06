# -*- coding: utf-8 -*-
"""
读取“表单二”土壤日均数据并清洗（带错误处理与结果输出）
"""
import pandas as pd
import json
from pathlib import Path
import sys

# ========= 可配置 =========
INPUT_FILE   = r"C:\Users\Raelon\OneDrive\文档\该地土壤湿度数据.xlsx"
SHEET_INDEX  = 1          # 表单二
MIN_OBS      = 40         # 质控阈值：半小时观测次数至少40/48
OUT_CSV      = "soil_daily_clean.csv"
REPORT_JSON  = "soil_daily_report.json"
MISS_RATE_CSV= "soil_missing_rate.csv"

REQUIRED_COLS = [
    "DATE","5cm_SM","10cm_SM","20cm_SM","40cm_SM",
    "5cm_ST","10cm_ST","20cm_ST","40cm_ST",
    "Number of nodes used to derive the daily mean of the sample area",
    "Number of observations used to summarize the half hour data into the daily mean"
]

RENAME_MAP = {
    "DATE":"date",
    "5cm_SM":"sm5", "10cm_SM":"sm10", "20cm_SM":"sm20", "40cm_SM":"sm40",
    "5cm_ST":"st5", "10cm_ST":"st10", "20cm_ST":"st20", "40cm_ST":"st40",
    "Number of nodes used to derive the daily mean of the sample area":"nodes",
    "Number of observations used to summarize the half hour data into the daily mean":"obs48"
}

KEEP_COLS = ["date","sm5","sm10","sm20","sm40","st5","st10","st20","st40","nodes","obs48"]

def fail(msg: str, code: int = 1):
    print(f"❌ 错误：{msg}")
    sys.exit(code)

def main():
    print("🚀 开始读取表单二（日均土壤数据）...")
    p = Path(INPUT_FILE)
    if not p.exists():
        fail(f"找不到文件：{INPUT_FILE}")

    # 读取
    try:
        df_raw = pd.read_excel(INPUT_FILE, sheet_name=SHEET_INDEX)
    except ValueError as e:
        fail(f"读取失败：{e}\n请检查工作表索引（当前={SHEET_INDEX}）或表名。")
    except Exception as e:
        fail(f"读取失败：{e}")

    print(f"✅ 成功读取：{len(df_raw)} 行，{len(df_raw.columns)} 列。")

    # 检查列
    missing = [c for c in REQUIRED_COLS if c not in df_raw.columns]
    if missing:
        print("⚠️  警告：缺少以下必需列（后续会尽量继续，但可能影响清洗质量）：")
        for c in missing: print("   -", c)
        print("ℹ️  当前列名清单：", list(df_raw.columns))

    # 重命名（仅能改到存在的列）
    rename_map_in_use = {k: v for k, v in RENAME_MAP.items() if k in df_raw.columns}
    df = df_raw.rename(columns=rename_map_in_use).copy()

    # 日期解析
    if "date" not in df.columns:
        fail("未找到日期列 DATE（或未被正确重命名为 date）。")

    print("🕒  解析日期...")
    date_before_na = df["date"].isna().sum()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    date_after_na = df["date"].isna().sum()
    if df["date"].notna().sum() == 0:
        fail("日期解析失败：所有日期均为 NaT，请检查日期格式（YYYY-MM-DD）。")

    # 丢弃无法解析日期的行
    dropped_date_na = int(date_after_na)
    if dropped_date_na > 0:
        print(f"⚠️  有 {dropped_date_na} 行日期无法解析，已剔除。")

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # 观测次数转数值
    if "obs48" in df.columns:
        df["obs48"] = pd.to_numeric(df["obs48"], errors="coerce")
    else:
        print("⚠️  未找到 obs48 列（半小时观测次数），将跳过基于 obs48 的质控。")

    # 质控：obs48 >= MIN_OBS
    removed_low_obs = 0
    if "obs48" in df.columns:
        before = len(df)
        df = df[df["obs48"] >= MIN_OBS].reset_index(drop=True)
        removed_low_obs = before - len(df)
        if removed_low_obs > 0:
            print(f"🧹  质控：剔除 obs48 < {MIN_OBS} 的天数 {removed_low_obs} 行。")

    # 只保留关键列（存在的才保留）
    keep_cols_in_use = [c for c in KEEP_COLS if c in df.columns]
    df_out = df[keep_cols_in_use].copy()

    # 缺失率统计（仅对输出列）
    miss_rate = (df_out.isna().sum() / len(df_out)).sort_values(ascending=False)
    miss_rate.to_csv(MISS_RATE_CSV, encoding="utf-8-sig")

    # 汇总报告
    report = {
        "input_rows": int(len(df_raw)),
        "output_rows": int(len(df_out)),
        "date_min": str(df_out["date"].min().date()) if "date" in df_out.columns and len(df_out) else None,
        "date_max": str(df_out["date"].max().date()) if "date" in df_out.columns and len(df_out) else None,
        "removed_date_parse_failed": int(dropped_date_na),
        "removed_low_obs": int(removed_low_obs),
        "kept_columns": keep_cols_in_use,
        "missing_required_columns": missing
    }
    Path(REPORT_JSON).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 导出
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

    print("\n🎉  清洗完成：")
    print(f"- 输出数据：{OUT_CSV}（{len(df_out)} 行）")
    print(f"- 缺失率表：{MISS_RATE_CSV}")
    print(f"- 清洗报告：{REPORT_JSON}")
    print("\n📌 摘要：")
    for k, v in report.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()

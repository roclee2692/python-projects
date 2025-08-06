# -*- coding: utf-8 -*-
"""
功能：
1) 读取"降水量等逐时气象数据.xls"
2) 统一列名到规范集：['time','T','Po','P','Pa','U','DD','Ff','ff10','ff3','N','WW','W1','W2','Tn','Tx','Cl','Nh','H','Cm','Ch','VV','Td','RRR','tR','5cm_SM']
3) 解析"当地时间"为 DatetimeIndex（Asia/Shanghai），升序
4) 同时刻去重（保留最后观测）
5) 负降水裁剪为0，极端降水上限150并记录数量
6) 输出清洗后的 q1_clean.csv 与缺失率报告 q1_missing_report.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import os

# ========= 可配置路径 =========
INPUT_FILE = r"C:\Users\Raelon\OneDrive\文档\降水量等逐时气象数据.xlsx"
OUT_CSV = "q1_clean.csv"
MISS_RPT = "q1_missing_report.csv"
LOG_JSON = "q1_clean_log.json"

# 检查输入文件是否存在
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"输入文件不存在: {INPUT_FILE}")

# 1) 读表
try:
    df_raw = pd.read_excel(INPUT_FILE, sheet_name=0, header=1)  # 如有多sheet可改
    print(f"成功读取Excel文件，原始数据形状: {df_raw.shape}")
    # 输出列名以便调试
    print(f"原始列名: {list(df_raw.columns)}")
except Exception as e:
    raise Exception(f"读取Excel文件失败: {str(e)}")

df = df_raw.copy()

# 2) 统一列名（大小写不敏感，去空格），按"别名→标准名"的映射来重命名
std_cols = [
    'time', 'T', 'Po', 'P', 'Pa', 'U', 'DD', 'Ff', 'ff10', 'ff3', 'N', 'WW', 'W1', 'W2',
    'Tn', 'Tx', 'Cl', 'Nh', 'H', 'Cm', 'Ch', 'VV', 'Td', 'RRR', 'tR', '5cm_SM'
]

aliases = {
    'time': ['time', '当地时间', '当地时间 长春市(机场)', '观测时间', '时刻', 'date', 'datetime', '日期', '时间'],
    'T': ['t', '气温', '气温(℃)'],
    'Po': ['po', '站压', '站点气压', '气压(站)'],
    'P': ['p', '海平面气压', '海压'],
    'Pa': ['pa', '气压趋势', '三小时气压变化', '3小时气压变化'],
    'U': ['u', '相对湿度', '相对湿度(%)'],
    'DD': ['dd', '风向', '风向(度)'],
    'Ff': ['ff', '平均风速', '风速', '风速(m/s)'],
    'ff10': ['ff10', '最大风10', '最大风(10分钟)', '10分钟最大风'],
    'ff3': ['ff3', '最大风3', 'Ff3', '3分钟最大风'],
    'N': ['n', '总云量'],
    'WW': ['ww', '现时天气', '天气现象'],
    'W1': ['w1', '过去天气1'],
    'W2': ['w2', '过去天气2'],
    'Tn': ['tn', '最低气温'],
    'Tx': ['tx', '最高气温'],
    'Cl': ['cl', '低云型', 'ci', 'c1'],
    'Nh': ['nh', '低云量', '云量c1'],
    'H': ['h', '低云高', '云底高度'],
    'Cm': ['cm', '中云型'],
    'Ch': ['ch', '高云型'],
    'VV': ['vv', '能见度'],
    'Td': ['td', '露点温度'],
    'RRR': ['rrr', '降水量', '小时降水', '降雨量'],
    'tR': ['tr', '达到规定降水量的时间', '降水时长'],
    '5cm_SM': ['5cm_sm', '5cm含水量', '土壤5cm含水量', 'soil_moisture_5cm']
}


# 构建 现有列 -> 标准列 的映射
def normalize(s: str) -> str:
    return str(s).strip().lower().replace(' ', '').replace('\u3000', '')


col_norm = {c: normalize(c) for c in df.columns}
rename_map = {}

# 打印规范化后的列名以便调试
print("规范化后的列名:")
for orig, norm in col_norm.items():
    print(f"  {orig} -> {norm}")

for std, alist in aliases.items():
    norm_aliases = [normalize(a) for a in alist]
    # 在现有列中找第一个匹配的别名
    for raw_col, norm_col in col_norm.items():
        if norm_col in norm_aliases:
            rename_map[raw_col] = std
            print(f"映射列: {raw_col} -> {std}")
            break

# 如果没有找到时间列，尝试更灵活的匹配
if 'time' not in rename_map.values():
    for col, norm_col in col_norm.items():
        # 检查列名是否包含"时间"或"日期"等关键词
        if '时间' in col or '日期' in col or 'time' in norm_col or 'date' in norm_col:
            rename_map[col] = 'time'
            print(f"通过关键词匹配到时间列: {col} -> time")
            break

# 应用重命名
df = df.rename(columns=rename_map)

# 提示未命中的关键列（不一定全都有）
missing_std = [c for c in std_cols if c not in df.columns]
if missing_std:
    print("提示：以下标准列未在原表中匹配到（不影响导出，可后续补充）：", missing_std)


# 3) 解析时间列，生成 DatetimeIndex（Asia/Shanghai）
# 可能的时间格式示例：'31.12.2021 23:00' 或 '2021-12-31 23:00'
def parse_time_series(s: pd.Series) -> pd.Series:
    # 先尝试 dayfirst=True
    t = pd.to_datetime(s, errors='coerce', dayfirst=True)
    # 对仍为 NaT 的，尝试常见格式
    mask = t.isna()
    if mask.any():
        # 尝试以显式格式解析（欧式 dd.mm.yyyy HH:MM）
        t2 = pd.to_datetime(s[mask], format='%d.%m.%Y %H:%M', errors='coerce')
        t.loc[mask] = t2
    return t


if 'time' not in df.columns:
    # 更详细的错误信息
    print("\n时间列未找到，提供更多调试信息:")
    print(f"Excel文件中的列: {list(df_raw.columns)}")
    print(f"规范化后的列名: {col_norm}")
    print(f"别名映射: {rename_map}")

    # 如果有第一行数据，打印出来看看是否有明显的时间列
    if len(df_raw) > 0:
        print("\n第一行数据:")
        for col, val in df_raw.iloc[0].items():
            print(f"  {col}: {val}")

        # 尝试自动找到可能的时间列
        for col in df_raw.columns:
            try:
                sample = str(df_raw[col].iloc[0])
                # 检查是否可能是日期时间格式
                if (('/' in sample or '-' in sample or '.' in sample) and
                        (':' in sample or any(c.isdigit() for c in sample))):
                    print(f"\n可能的时间列: {col}, 样本值: {sample}")
                    # 尝试将该列作为时间列
                    df['time'] = df_raw[col]
                    print(f"已尝试将 {col} 设为时间列")
                    break
            except:
                continue

    # 如果仍未找到时间列，引发错误
    if 'time' not in df.columns:
        raise ValueError("未找到时间列，请检查时间列别名映射。您可能需要手动指定时间列或修改别名映射。")

t = parse_time_series(df['time'])
if t.isna().all():
    raise ValueError("时间解析失败：请检查时间格式（例如是否为 'dd.mm.yyyy HH:MM' 或 'yyyy-mm-dd HH:MM'）。")

# 设为索引并本地化时区（不做转换，仅标注为 Asia/Shanghai）
df = df.assign(time=t).dropna(subset=['time']).sort_values('time')
df = df.set_index('time')
try:
    df.index = df.index.tz_localize('Asia/Shanghai', nonexistent='shift_forward', ambiguous='NaT')
except Exception:
    # 已有时区或个别异常，忽略本地化
    pass

# 4) 同一时刻去重（保留最后观测）
before = len(df)
df = df[~df.index.duplicated(keep='last')]
after = len(df)
duplicates_removed = before - after

# 5) 数值列转数值（文本转 NaN），并统计缺失
num_like_cols = [c for c in df.columns if c.lower() not in ['ww', 'w1', 'w2', 'dd', 'cl', 'cm', 'ch']]
for c in num_like_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# 6) 降水裁剪（负数→0，上限150）
if 'RRR' in df.columns:
    neg_count = (df['RRR'] < 0).sum(skipna=True)
    over_count = (df['RRR'] > 150).sum(skipna=True)
    df['RRR'] = df['RRR'].clip(lower=0, upper=150)
else:
    neg_count = over_count = 0

# 7) 缺失率报告
miss_rate = (df.isna().sum() / len(df)).sort_values(ascending=False)
miss_rate.to_csv(MISS_RPT, encoding='utf-8-sig')

# 8) 导出清洗后的逐时数据
# 仅导出标准列中实际存在的列，避免写出无关列
export_cols = ['T', 'Po', 'P', 'Pa', 'U', 'DD', 'Ff', 'ff10', 'ff3', 'N', 'WW', 'W1', 'W2',
               'Tn', 'Tx', 'Cl', 'Nh', 'H', 'Cm', 'Ch', 'VV', 'Td', 'RRR', 'tR', '5cm_SM']
export_cols = [c for c in export_cols if c in df.columns]
df_out = df[export_cols].copy()
df_out.to_csv(OUT_CSV, encoding='utf-8-sig', date_format='%Y-%m-%d %H:%M')

# 9) 记录日志
log = {
    "input_rows": int(len(df_raw)),
    "output_rows": int(len(df_out)),
    "duplicates_removed": int(duplicates_removed),
    "rrr_neg_clipped": int(neg_count),
    "rrr_upper_clipped": int(over_count),
    "missing_columns_not_found": missing_std
}
Path(LOG_JSON).write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding='utf-8')

print("✅ 清洗完成：")
print(f"- 输出文件：{OUT_CSV}")
print(f"- 缺失率报告：{MISS_RPT}")
print(f"- 清洗日志：{LOG_JSON}")
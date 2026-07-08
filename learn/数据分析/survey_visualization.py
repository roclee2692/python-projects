"""
大学生编程环境配置困难调查数据可视化
项目：码上修——校园编程环境配置与数字学习服务站
数据来源：本校问卷调查（168份有效问卷）+ Stack Overflow 2024 + JetBrains 2024
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import sys

# Windows 控制台 UTF-8 输出
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ── 中文字体配置 ──────────────────────────────────────────────
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 全局配色方案 ──────────────────────────────────────────────
C = {
    "blue":   ["#1a73e8", "#4a9af5", "#7ab8f5", "#a8d1f5", "#d4e8fa"],
    "warm":   ["#e74c3c", "#f39c12", "#2ecc71", "#3498db", "#9b59b6", "#1abc9c"],
    "pie":    ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"],
    "bar":    ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"],
    "green":  ["#27ae60", "#2ecc71", "#58d68d", "#85e89d"],
    "red":    ["#c0392b", "#e74c3c", "#f1948a", "#fadbd8"],
}

# 输出目录
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)


def _clean(ax):
    """去除上、右边框"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _barh_label(ax, bars, fmt="{:.1f}%"):
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.8, bar.get_y() + bar.get_height() / 2,
                fmt.format(w), va="center", fontsize=10, fontweight="bold")


def _bar_label(ax, bars, fmt="{:.1f}%"):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                fmt.format(h), ha="center", fontsize=10, fontweight="bold")


# ══════════════════════════════════════════════════════════════
# 图1：问卷调查概览
# ══════════════════════════════════════════════════════════════
def plot_01_overview():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("问卷调查基本概况", fontsize=16, fontweight="bold", y=1.05)
    cards = [
        ("发放问卷", "190", "份", C["blue"][0]),
        ("有效回收", "168", "份", C["green"][0]),
        ("有效回收率", "88.4", "%", C["warm"][0]),
        ("调研周期", "7", "天", C["warm"][3]),
    ]
    for ax, (label, val, unit, color) in zip(axes, cards):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.58, val, ha="center", va="center",
                fontsize=38, fontweight="bold", color=color)
        ax.text(0.5, 0.3, unit, ha="center", va="center", fontsize=14, color="#555")
        ax.text(0.5, 0.08, label, ha="center", va="center",
                fontsize=13, fontweight="bold", color="#333")
        ax.axis("off")
        ax.add_patch(plt.Circle((0.5, 0.5), 0.38, fill=False,
                                edgecolor=color, linewidth=2.5, alpha=0.35))
    plt.tight_layout()
    plt.savefig(f"{OUT}/01_问卷概览.png", dpi=150, bbox_inches="tight")
    plt.show(); print("[OK] 图1 完成")


# ══════════════════════════════════════════════════════════════
# 图2：年级 & 操作系统分布
# ══════════════════════════════════════════════════════════════
def plot_02_demographics():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("受访学生基本情况", fontsize=16, fontweight="bold", y=1.02)

    # 年级
    w1, t1, a1 = ax1.pie(
        [67.3, 32.7], labels=["大一", "大二"], autopct="%1.1f%%",
        startangle=90, colors=C["pie"][:2],
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        pctdistance=0.55, textprops={"fontsize": 13})
    for t in a1: t.set_fontweight("bold"); t.set_fontsize(12)
    ax1.set_title("年级分布", fontsize=14, fontweight="bold", pad=15)

    # 操作系统
    w2, t2, a2 = ax2.pie(
        [92.9, 7.1], labels=["Windows", "Mac"], autopct="%1.1f%%",
        startangle=90, colors=["#2196F3", "#FF9800"],
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        pctdistance=0.55, textprops={"fontsize": 13})
    for t in a2: t.set_fontweight("bold"); t.set_fontsize(12)
    ax2.set_title("操作系统分布", fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(f"{OUT}/02_年级与系统.png", dpi=150, bbox_inches="tight")
    plt.show(); print("[OK] 图2 完成")


# ══════════════════════════════════════════════════════════════
# 图3：编程环境配置困难总览
# ══════════════════════════════════════════════════════════════
def plot_03_env_difficulties():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("编程环境配置困难情况", fontsize=16, fontweight="bold", y=1.02)

    # 左：主要问题比例
    labels = [
        "曾出现环境配置错误\n(PATH/MinGW/VS Code等)",
        "多次反复出现\n类似配置错误",
        "pip安装第三方库\n超时/版本冲突/报错",
        "能独立看懂英文\n报错信息并排错",
    ]
    vals   = [77.4, 43.5, 69.6, 18.5]
    colors = [C["red"][1], C["warm"][1], C["red"][0], C["green"][0]]
    bars = ax1.barh(np.arange(len(labels)), vals, color=colors,
                    edgecolor="white", linewidth=1.2, height=0.6)
    ax1.set_yticks(np.arange(len(labels)))
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlim(0, 95); ax1.invert_yaxis(); _clean(ax1)
    _barh_label(ax1, bars)
    ax1.set_title("环境配置问题比例", fontsize=13, fontweight="bold", pad=10)

    # 右：学习阻碍因素
    obs = ["环境配置问题", "代码语法Bug", "编程逻辑思路"]
    ovals = [62.5, 23.2, 14.3]
    w, t, a = ax2.pie(
        ovals, labels=obs, autopct="%1.1f%%", startangle=140,
        colors=[C["red"][0], C["warm"][1], C["blue"][0]],
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        pctdistance=0.6, labeldistance=1.15, textprops={"fontsize": 12})
    for tt in a: tt.set_fontweight("bold"); tt.set_fontsize(11)
    ax2.set_title("编程学习主要阻碍因素", fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(f"{OUT}/03_环境配置困难.png", dpi=150, bbox_inches="tight")
    plt.show(); print("[OK] 图3 完成")


# ══════════════════════════════════════════════════════════════
# 图4：Python / C++ 分语言痛点
# ══════════════════════════════════════════════════════════════
def plot_04_language_issues():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("分语言编程环境痛点", fontsize=16, fontweight="bold", y=1.02)

    # Python
    py_l = ["遇到pip安装问题", "超时/下载慢", "版本冲突报错", "命令行报错看不懂", "能独立解决"]
    py_v = [69.6, 45.0, 38.0, 51.1, 18.5]
    py_c = [C["red"][0], C["warm"][1], C["warm"][0], C["red"][1], C["green"][0]]
    bars1 = ax1.barh(np.arange(len(py_l)), py_v, color=py_c,
                     edgecolor="white", linewidth=1.2, height=0.55)
    ax1.set_yticks(np.arange(len(py_l)))
    ax1.set_yticklabels(py_l, fontsize=10)
    ax1.set_xlim(0, 82); ax1.invert_yaxis(); _clean(ax1)
    _barh_label(ax1, bars1)
    ax1.set_title("Python 第三方库安装问题", fontsize=13, fontweight="bold", pad=10)

    # C/C++
    cpp_l = ["不会区分32/64位编译器", "不清楚配置系统环境变量",
             "编译/运行失败", "VS Code无法识别编译器"]
    cpp_v = [65.5, 65.5, 52.0, 40.0]
    cpp_c = [C["blue"][i] for i in range(4)]
    bars2 = ax2.barh(np.arange(len(cpp_l)), cpp_v, color=cpp_c,
                     edgecolor="white", linewidth=1.2, height=0.55)
    ax2.set_yticks(np.arange(len(cpp_l)))
    ax2.set_yticklabels(cpp_l, fontsize=10)
    ax2.set_xlim(0, 80); ax2.invert_yaxis(); _clean(ax2)
    _barh_label(ax2, bars2)
    ax2.set_title("C/C++ 编译器配置问题", fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig(f"{OUT}/04_分语言痛点.png", dpi=150, bbox_inches="tight")
    plt.show(); print("[OK] 图4 完成")


# ══════════════════════════════════════════════════════════════
# 图5：求助渠道
# ══════════════════════════════════════════════════════════════
def plot_05_help_channels():
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("遇到环境配置问题后的求助渠道", fontsize=16, fontweight="bold", y=1.01)

    ch  = ["百度/B站/CSDN\n搜索教程", "询问同学\n或高年级学长",
           "咨询任课老师", "送到校外\n维修店付费", "暂时搁置/\n借用他人电脑"]
    cv  = [53.6, 32.1, 8.3, 3.0, 3.0]
    cc  = C["bar"][:5]
    bars = ax.bar(ch, cv, color=cc, edgecolor="white", linewidth=1.5, width=0.6)
    _bar_label(ax, bars)
    ax.set_ylabel("占比 (%)"); ax.set_ylim(0, 64); _clean(ax)

    ax.annotate("超过80%依赖网络教程和同学互助",
                xy=(0.5, 0.92), xycoords="axes fraction",
                fontsize=11, ha="center", color="#e74c3c", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fff3e0", ec="#e74c3c", alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{OUT}/05_求助渠道.png", dpi=150, bbox_inches="tight")
    plt.show(); print("[OK] 图5 完成")


# ══════════════════════════════════════════════════════════════
# 图6：电脑轻度维护需求
# ══════════════════════════════════════════════════════════════
def plot_06_maintenance():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("电脑轻度维护需求情况", fontsize=16, fontweight="bold", y=1.02)

    # 左
    ml = ["至少存在一项问题", "开机卡顿", "C盘空间不足", "会定期清理维护"]
    mv = [64.9, 42.0, 38.0, 21.4]
    mc = [C["warm"][0], C["warm"][1], C["red"][1], C["green"][0]]
    bars1 = ax1.barh(np.arange(len(ml)), mv, color=mc,
                     edgecolor="white", linewidth=1.2, height=0.5)
    ax1.set_yticks(np.arange(len(ml)))
    ax1.set_yticklabels(ml, fontsize=11)
    ax1.set_xlim(0, 80); ax1.invert_yaxis(); _clean(ax1)
    _barh_label(ax1, bars1)
    ax1.set_title("日常电脑问题", fontsize=13, fontweight="bold", pad=10)

    # 右
    rl = ["下载非正规安装包\n附带捆绑软件", "会定期清理维护"]
    rv = [37.5, 21.4]
    rc = [C["red"][0], C["green"][0]]
    w, t, a = ax2.pie(
        rv, labels=rl, autopct="%1.1f%%", startangle=90, colors=rc,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        pctdistance=0.55, labeldistance=1.2, textprops={"fontsize": 11})
    for tt in a: tt.set_fontweight("bold")
    ax2.set_title("软件安全与维护意识", fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(f"{OUT}/06_电脑维护需求.png", dpi=150, bbox_inches="tight")
    plt.show(); print("[OK] 图6 完成")


# ══════════════════════════════════════════════════════════════
# 图7：付费意愿 & 价格偏好
# ══════════════════════════════════════════════════════════════
def plot_07_payment():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("校园付费服务接受意愿与价格偏好", fontsize=16, fontweight="bold", y=1.02)

    # 左
    wl, wv, wc = ["愿意尝试", "中立", "不愿意"], [83.3, 13.1, 3.6], [C["green"][0], C["warm"][1], C["red"][0]]
    w, t, a = ax1.pie(
        wv, labels=wl, autopct="%1.1f%%", startangle=90, colors=wc,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        pctdistance=0.6, labeldistance=1.2, textprops={"fontsize": 12})
    for tt in a: tt.set_fontweight("bold"); tt.set_fontsize(11)
    ax1.set_title("服务接受意愿", fontsize=13, fontweight="bold", pad=15)

    # 右
    pl = ["5~10元", "11~15元", "16~20元+"]
    pv = [61.9, 28.6, 9.5]
    pc = [C["green"][0], C["blue"][0], C["warm"][3]]
    bars2 = ax2.bar(pl, pv, color=pc, edgecolor="white", linewidth=1.5, width=0.5)
    _bar_label(ax2, bars2)
    ax2.set_ylabel("占比 (%)"); ax2.set_ylim(0, 75); _clean(ax2)
    ax2.set_title("单次全套环境安装调试\n心理价位", fontsize=13, fontweight="bold", pad=10)
    ax2.annotate("学生对价格较为敏感\n初期应采用低价模式",
                xy=(0.5, 0.92), xycoords="axes fraction",
                fontsize=10, ha="center", color="#e74c3c", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fff3e0", ec="#e74c3c", alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{OUT}/07_付费意愿.png", dpi=150, bbox_inches="tight")
    plt.show(); print("[OK] 图7 完成")


# ══════════════════════════════════════════════════════════════
# 图8：期望的附赠服务
# ══════════════════════════════════════════════════════════════
def plot_08_services():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle("学生期望的附赠服务内容", fontsize=16, fontweight="bold", y=1.01)

    sl = ["基础操作讲解文档", "常见报错避坑指南", "1次短期线上答疑", "学期优惠打包服务"]
    sv = [72.6, 68.5, 57.1, 66.7]
    sc = [C["blue"][0], C["blue"][1], C["blue"][2], C["warm"][3]]
    bars = ax.barh(np.arange(len(sl)), sv, color=sc,
                   edgecolor="white", linewidth=1.5, height=0.55)
    ax.set_yticks(np.arange(len(sl)))
    ax.set_yticklabels(sl, fontsize=12)
    ax.set_xlim(0, 85); ax.invert_yaxis(); _clean(ax)
    _barh_label(ax, bars)
    ax.set_title("「环境配置 + 操作说明 + 短期答疑」模式受青睐", fontsize=12, pad=10)

    plt.tight_layout()
    plt.savefig(f"{OUT}/08_期望服务.png", dpi=150, bbox_inches="tight")
    plt.show(); print("[OK] 图8 完成")


# ══════════════════════════════════════════════════════════════
# 图9：学生主要顾虑
# ══════════════════════════════════════════════════════════════
def plot_09_concerns():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle("选择校园数字服务时的主要顾虑", fontsize=16, fontweight="bold", y=1.01)

    cl = ["电脑文件隐私泄露", "服务技术不专业\n越修问题越多", "缺少售后保障\n二次问题无人负责"]
    cv = [74.4, 63.1, 58.9]
    cc = [C["red"][0], C["warm"][1], C["warm"][0]]
    bars = ax.barh(np.arange(len(cl)), cv, color=cc,
                   edgecolor="white", linewidth=1.5, height=0.5)
    ax.set_yticks(np.arange(len(cl)))
    ax.set_yticklabels(cl, fontsize=12)
    ax.set_xlim(0, 95); ax.invert_yaxis(); _clean(ax)
    _barh_label(ax, bars)

    strats = [
        "→ 操作前备份提醒 + 用户在场 + 不查看私人文件",
        "→ 建立常见问题库 + 标准化操作流程",
        "→ 24h内一次免费复查",
    ]
    for i, s in enumerate(strats):
        ax.text(78, i + 0.3, s, fontsize=8.5, color="#2196F3", style="italic")
    ax.set_title("隐私保护、技术可靠性、售后保障是三大核心关切", fontsize=12, pad=10)

    plt.tight_layout()
    plt.savefig(f"{OUT}/09_学生顾虑.png", dpi=150, bbox_inches="tight")
    plt.show(); print("[OK] 图9 完成")


# ══════════════════════════════════════════════════════════════
# 图10：本校 vs 全球数据对比
# ══════════════════════════════════════════════════════════════
def plot_10_comparison():
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("编程环境配置困难：本校调查 vs 全球数据对比",
                 fontsize=16, fontweight="bold", y=1.01)

    cats = ["环境配置\n是主要障碍", "依赖管理/\n版本冲突",
            "每天>30min\n搜索方案", "能独立\n看懂报错", "AI工具\n辅助编程"]
    local  = [62.5, 69.6, 61.0, 18.5, 30.0]
    global_ = [45.0, 55.0, 61.0, 45.0, 49.0]

    x = np.arange(len(cats)); w = 0.32
    b1 = ax.bar(x - w/2, local,  w, label="本校调查 (n=168)",
                color="#e74c3c", edgecolor="white", linewidth=1.2)
    b2 = ax.bar(x + w/2, global_, w, label="全球数据 (SO/JB 2024)",
                color="#3498db", edgecolor="white", linewidth=1.2)
    _bar_label(ax, b1, fmt="{:.0f}%"); _bar_label(ax, b2, fmt="{:.0f}%")
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=10)
    ax.set_ylabel("占比 (%)"); ax.set_ylim(0, 82)
    ax.legend(fontsize=11, loc="upper right"); _clean(ax)

    ax.annotate("本校学生能独立排错的比例\n远低于全球水平",
                xy=(3, 18.5), xytext=(3.8, 38),
                fontsize=10, color="#c0392b", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#c0392b", alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{OUT}/10_本校vs全球.png", dpi=150, bbox_inches="tight")
    plt.show(); print("[OK] 图10 完成")


# ══════════════════════════════════════════════════════════════
# 图11：Stack Overflow 2024 编程语言排名
# ══════════════════════════════════════════════════════════════
def plot_11_so_languages():
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Stack Overflow 2024：最受欢迎编程语言",
                 fontsize=16, fontweight="bold", y=1.01)

    langs = ["JavaScript", "HTML/CSS", "Python", "SQL", "TypeScript",
             "Bash/Shell", "Java", "C#", "C++", "C", "PHP", "Rust"]
    usage = [62.3, 52.9, 51.0, 51.5, 38.5, 33.9, 30.3, 27.1, 23.0, 20.0, 18.2, 12.6]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(langs)))

    bars = ax.barh(np.arange(len(langs)), usage, color=colors,
                   edgecolor="white", linewidth=1.2, height=0.65)
    ax.set_yticks(np.arange(len(langs)))
    ax.set_yticklabels(langs, fontsize=11)
    ax.set_xlim(0, 75); ax.invert_yaxis(); _clean(ax)
    _barh_label(ax, bars)
    ax.set_title("全球 65,000+ 开发者调查", fontsize=11, pad=10)

    plt.tight_layout()
    plt.savefig(f"{OUT}/11_SO语言排名.png", dpi=150, bbox_inches="tight")
    plt.show(); print("[OK] 图11 完成")


# ══════════════════════════════════════════════════════════════
# 图12：JetBrains 2024 语言趋势 & AI 工具
# ══════════════════════════════════════════════════════════════
def plot_12_jetbrains():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("JetBrains Developer Ecosystem 2024", fontsize=16, fontweight="bold", y=1.02)

    # 左：语言 2017→2024
    langs = ["JavaScript", "Python", "TypeScript", "Java", "C#", "Go", "Rust"]
    v17 = [65, 32, 12, 45, 30, 8, 3]
    v24 = [61, 53, 35, 38, 28, 14, 12]
    x = np.arange(len(langs)); w = 0.32
    ax1.bar(x - w/2, v17, w, label="2017", color="#bdc3c7", edgecolor="white")
    ax1.bar(x + w/2, v24, w, label="2024", color="#2196F3", edgecolor="white")
    ax1.set_xticks(x); ax1.set_xticklabels(langs, fontsize=10, rotation=20)
    ax1.set_ylabel("使用率 (%)"); ax1.set_ylim(0, 80)
    ax1.legend(fontsize=10); _clean(ax1)
    ax1.set_title("语言使用率变化 (2017→2024)", fontsize=13, fontweight="bold", pad=10)
    ax1.annotate("+21%", xy=(1 + w/2, 53), xytext=(1.6, 65),
                fontsize=11, color="#27ae60", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#27ae60"))
    ax1.annotate("+23%", xy=(2 + w/2, 35), xytext=(2.6, 50),
                fontsize=11, color="#27ae60", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#27ae60"))

    # 右：AI 工具
    tl = ["ChatGPT\n(尝试过)", "ChatGPT\n(经常使用)", "Copilot\n(尝试过)", "Copilot\n(经常使用)"]
    tv = [69, 49, 40, 26]
    tc = ["#4CAF50", "#2E7D32", "#2196F3", "#1565C0"]
    bars = ax2.bar(tl, tv, color=tc, edgecolor="white", linewidth=1.5, width=0.5)
    _bar_label(ax2, bars, fmt="{:.0f}%")
    ax2.set_ylabel("占比 (%)"); ax2.set_ylim(0, 80); _clean(ax2)
    ax2.set_title("AI 编程工具采用率", fontsize=13, fontweight="bold", pad=10)
    ax2.annotate("80%公司允许使用AI工具\n仅11%完全禁止",
                xy=(0.5, 0.92), xycoords="axes fraction",
                fontsize=10, ha="center", color="#FF9800", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fff8e1", ec="#FF9800", alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{OUT}/12_JetBrains趋势.png", dpi=150, bbox_inches="tight")
    plt.show(); print("[OK] 图12 完成")


# ══════════════════════════════════════════════════════════════
# 图13：汇总仪表盘
# ══════════════════════════════════════════════════════════════
def plot_13_dashboard():
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("「码上修」大学生编程环境配置困难调查 — 汇总仪表盘",
                 fontsize=20, fontweight="bold", y=0.98)

    # ── 顶行：5个指标卡 ──
    cards = [
        ("有效问卷", "168份",   C["blue"][0]),
        ("环境配置遇困", "77.4%", C["red"][0]),
        ("pip安装问题", "69.6%",  C["warm"][0]),
        ("环境配置是\n主要障碍", "62.5%", C["red"][1]),
        ("愿意付费", "83.3%",    C["green"][0]),
    ]
    for i, (lab, val, col) in enumerate(cards):
        ax = fig.add_axes([0.02 + i * 0.196, 0.82, 0.18, 0.12])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.58, val, ha="center", va="center",
                fontsize=28, fontweight="bold", color=col)
        ax.text(0.5, 0.15, lab, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#333")
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False,
                                   edgecolor=col, linewidth=2.5, alpha=0.5,
                                   transform=ax.transAxes))

    # ── 中左：求助渠道 ──
    ax_ch = fig.add_axes([0.06, 0.48, 0.42, 0.3])
    ch_l = ["网络教程", "同学互助", "老师", "维修店", "搁置"]
    ch_v = [53.6, 32.1, 8.3, 3.0, 3.0]
    bars = ax_ch.barh(np.arange(5), ch_v, color=C["bar"][:5],
                      edgecolor="white", linewidth=1, height=0.6)
    ax_ch.set_yticks(np.arange(5))
    ax_ch.set_yticklabels(ch_l, fontsize=10)
    ax_ch.invert_yaxis(); ax_ch.set_xlim(0, 65)
    _clean(ax_ch)
    for bar, val in zip(bars, ch_v):
        ax_ch.text(val + 1, bar.get_y() + bar.get_height() / 2,
                  f"{val}%", va="center", fontsize=10, fontweight="bold")
    ax_ch.set_title("求助渠道分布", fontsize=13, fontweight="bold")

    # ── 中右：学习障碍 ──
    ax_obs = fig.add_axes([0.56, 0.48, 0.4, 0.3])
    w, t, a = ax_obs.pie(
        [62.5, 23.2, 14.3], labels=["环境配置", "语法Bug", "编程逻辑"],
        autopct="%1.1f%%", startangle=140,
        colors=[C["red"][0], C["warm"][1], C["blue"][0]],
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        pctdistance=0.6, labeldistance=1.2, textprops={"fontsize": 11})
    for tt in a: tt.set_fontweight("bold")
    ax_obs.set_title("编程学习主要障碍", fontsize=13, fontweight="bold")

    # ── 底左：价格 ──
    ax_pr = fig.add_axes([0.06, 0.1, 0.42, 0.32])
    bars_p = ax_pr.bar(["5~10元", "11~15元", "16~20元+"],
                       [61.9, 28.6, 9.5],
                       color=[C["green"][0], C["blue"][0], C["warm"][3]],
                       edgecolor="white", linewidth=1.5, width=0.5)
    for bar, val in zip(bars_p, [61.9, 28.6, 9.5]):
        ax_pr.text(bar.get_x() + bar.get_width() / 2, val + 1,
                  f"{val}%", ha="center", fontsize=11, fontweight="bold")
    ax_pr.set_ylim(0, 75); _clean(ax_pr)
    ax_pr.set_title("单次服务心理价位", fontsize=13, fontweight="bold")

    # ── 底右：顾虑 ──
    ax_co = fig.add_axes([0.56, 0.1, 0.4, 0.32])
    bars_c = ax_co.barh(np.arange(3), [74.4, 63.1, 58.9],
                        color=[C["red"][0], C["warm"][1], C["warm"][0]],
                        edgecolor="white", linewidth=1, height=0.5)
    ax_co.set_yticks(np.arange(3))
    ax_co.set_yticklabels(["隐私泄露", "技术不专业", "缺少售后"], fontsize=11)
    ax_co.invert_yaxis(); ax_co.set_xlim(0, 90); _clean(ax_co)
    for bar, val in zip(bars_c, [74.4, 63.1, 58.9]):
        ax_co.text(val + 1, bar.get_y() + bar.get_height() / 2,
                  f"{val}%", va="center", fontsize=11, fontweight="bold")
    ax_co.set_title("学生主要顾虑", fontsize=13, fontweight="bold")

    plt.savefig(f"{OUT}/13_汇总仪表盘.png", dpi=150, bbox_inches="tight")
    plt.show(); print("[OK] 图13 汇总仪表盘完成")


# ══════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("  大学生编程环境配置困难调查 — 数据可视化")
    print("=" * 50)

    plot_01_overview()       # 问卷概览
    plot_02_demographics()   # 年级与系统
    plot_03_env_difficulties()  # 环境配置困难
    plot_04_language_issues()   # Python/C++ 痛点
    plot_05_help_channels()     # 求助渠道
    plot_06_maintenance()       # 电脑维护
    plot_07_payment()           # 付费意愿
    plot_08_services()          # 期望服务
    plot_09_concerns()          # 学生顾虑
    plot_10_comparison()        # 本校 vs 全球
    plot_11_so_languages()      # SO 语言排名
    plot_12_jetbrains()         # JetBrains 趋势
    plot_13_dashboard()         # 汇总仪表盘

    print("\n" + "=" * 50)
    print(f"  全部13张图表已保存至 {OUT}/ 目录")
    print("=" * 50)

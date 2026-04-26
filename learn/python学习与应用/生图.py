import os
import shutil
from graphviz import Digraph

# ===== Graphviz 路径 =====
graphviz_bin = r"C:\Program Files\Graphviz\bin"
os.environ["PATH"] = graphviz_bin + os.pathsep + os.environ["PATH"]

print("dot path:", shutil.which("dot"))

# ===== 输出到脚本所在目录 =====
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "enterprise_profile_radial")

# ===== 创建图 =====
dot = Digraph("EnterpriseProfileRadial", format="png")
dot.engine = "twopi"   # 放射状布局
dot.attr(overlap="false", splines="true", sep="0.8")

# 节点样式
dot.attr(
    "node",
    shape="ellipse",
    style="solid",
    fontname="Microsoft YaHei",
    fontsize="14",
    color="black"
)

# 边样式
dot.attr(
    "edge",
    fontname="Microsoft YaHei",
    fontsize="12",
    color="black"
)

# ===== 中心节点 =====
dot.node("center", "企业画像", shape="ellipse", fontsize="18")

# ===== 一级模块 =====
modules = {
    "qual": "公司资质",
    "perf": "过往业绩",
    "person": "人员能力",
    "finance": "财务与信誉",
    "tags": "核心优势标签"
}

for node_id, label in modules.items():
    dot.node(node_id, label)
    dot.edge("center", node_id)

# ===== 二级字段 =====
# 公司资质
qual_fields = ["资质等级", "安全生产许可证", "营业执照", "资质匹配度"]
for i, label in enumerate(qual_fields, 1):
    node_id = f"qual_{i}"
    dot.node(node_id, label)
    dot.edge("qual", node_id)

# 过往业绩
perf_fields = ["项目类型", "项目金额", "完成年限", "总包/分包", "工程质量"]
for i, label in enumerate(perf_fields, 1):
    node_id = f"perf_{i}"
    dot.node(node_id, label)
    dot.edge("perf", node_id)

# 人员能力
person_fields = ["项目经理", "项目总工", "人员证书", "在岗情况"]
for i, label in enumerate(person_fields, 1):
    node_id = f"person_{i}"
    dot.node(node_id, label)
    dot.edge("person", node_id)

# 财务与信誉
finance_fields = ["财务指标", "信用等级", "失信/处罚记录", "银行授信/营运能力"]
for i, label in enumerate(finance_fields, 1):
    node_id = f"finance_{i}"
    dot.node(node_id, label)
    dot.edge("finance", node_id)

# 核心优势标签
tag_fields = ["水利项目经验", "机电安装能力", "大额业绩", "一级资质", "区域匹配"]
for i, label in enumerate(tag_fields, 1):
    node_id = f"tags_{i}"
    dot.node(node_id, label)
    dot.edge("tags", node_id)

# ===== 导出 =====
dot.render(output_path, cleanup=True)
print("生成完成：", output_path + ".png")
from pathlib import Path

from PIL import Image, ImageDraw

# 画布参数（像素）
WIDTH, HEIGHT = 900, 700
OFFSET_X, OFFSET_Y = 450, 350  # 将“数学坐标”映射到图像坐标

# 重试参数：整体放大并上移，主体更接近参考图占比
SCENE_SCALE = 1.20
SCENE_SHIFT_X = -8
SCENE_SHIFT_Y = 58

# 场景元素缓存（用于同时导出 SVG）
svg_elements = []


def to_img_xy(x: float, y: float) -> tuple[float, float]:
    """坐标转换：数学坐标 -> 图像坐标（Y 轴翻转）。"""
    tx = x * SCENE_SCALE + SCENE_SHIFT_X
    ty = y * SCENE_SCALE + SCENE_SHIFT_Y
    return tx + OFFSET_X, OFFSET_Y - ty


def add_rect(draw: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float, fill: str, border: str | None = None):
    edge = border if border else fill

    # 数学坐标里 (x, y) 是左上角；图像坐标里要先转两个角点
    x1, y1 = to_img_xy(x, y)
    x2, y2 = to_img_xy(x + w, y - h)
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))

    draw.rectangle([left, top, right, bottom], fill=fill, outline=edge, width=1)
    svg_elements.append(
        f'  <rect x="{left:.2f}" y="{top:.2f}" width="{right-left:.2f}" height="{bottom-top:.2f}" '
        f'fill="{fill}" stroke="{edge}" stroke-width="1"/>'
    )


def add_line(draw: ImageDraw.ImageDraw, x1: float, y1: float, x2: float, y2: float, color: str, size: int = 2):
    p1 = to_img_xy(x1, y1)
    p2 = to_img_xy(x2, y2)
    draw.line([p1, p2], fill=color, width=size)
    svg_elements.append(
        f'  <line x1="{p1[0]:.2f}" y1="{p1[1]:.2f}" x2="{p2[0]:.2f}" y2="{p2[1]:.2f}" '
        f'stroke="{color}" stroke-width="{size}"/>'
    )


def add_circle(draw: ImageDraw.ImageDraw, x: float, y: float, r: float, fill: str, border: str | None = None):
    edge = border if border else fill
    cx, cy = to_img_xy(x, y)
    box = [cx - r, cy - r, cx + r, cy + r]
    draw.ellipse(box, fill=fill, outline=edge, width=1)
    svg_elements.append(
        f'  <circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" stroke="{edge}" stroke-width="1"/>'
    )


def draw_scene_base(draw: ImageDraw.ImageDraw):
    add_rect(draw, -450, -210, 900, 160, "#7cab69")
    add_rect(draw, -450, -234, 900, 24, "#607653")


def draw_left_main_tower(draw: ImageDraw.ImageDraw):
    top_y = 238
    floor_h = 90
    gap = 8
    for i in range(5):
        y = top_y - i * (floor_h + gap)
        x = -152 - i * 4
        w = 172 + i * 8
        add_rect(draw, x, y, w, floor_h, "#8a3f40", "#6d2c2d")
        add_rect(draw, x + 8, y - 6, w - 16, floor_h - 16, "#707f88", "#5a6770")
        add_line(draw, x + 12, y - 10, x + w - 14, y - floor_h + 18, "#53616b", 2)


def draw_yellow_shaft(draw: ImageDraw.ImageDraw):
    x = 32
    top = 262
    h = 514
    w = 42
    add_rect(draw, x, top, w, h, "#f1c300", "#d4a900")
    for row in range(24):
        y = top - 18 - row * 20
        add_circle(draw, x + 12, y, 1.8, "#907800")
        add_circle(draw, x + 30, y, 1.8, "#907800")
    add_rect(draw, x + 4, top + 4, w - 8, 5, "#f6cf41", "#f6cf41")


def draw_right_side_tower(draw: ImageDraw.ImageDraw):
    x = 74
    top = 246
    w = 78
    h = 498
    add_rect(draw, x, top, w, h, "#4f5963", "#3e4650")
    for i in range(1, 5):
        y = top - i * 100
        add_line(draw, x, y, x + w, y, "#404953", 2)
    for row in range(20):
        wy = top - 18 - row * 24
        for col in range(2):
            wx = x + 13 + col * 31
            add_rect(draw, wx, wy, 14, 11, "#47505a", "#39414a")


def draw_roof_frame(draw: ImageDraw.ImageDraw):
    add_rect(draw, -150, 252, 170, 14, "#7d3435", "#7d3435")
    add_line(draw, -136, 266, -136, 250, "#7d3435", 3)
    add_line(draw, -45, 266, -45, 250, "#7d3435", 3)


def draw_foreground_trees(draw: ImageDraw.ImageDraw):
    tree_positions = [-300, -250, -200, -150, -100, -55, -10, 35, 230, 275, 320, 360]
    for x in tree_positions:
        add_rect(draw, x - 4, -212, 8, 26, "#5b4634")
        add_circle(draw, x - 18, -198, 24, "#2f703c")
        add_circle(draw, x + 2, -188, 26, "#327741")
        add_circle(draw, x + 20, -199, 22, "#2c6b39")


def draw_connector_beams(draw: ImageDraw.ImageDraw):
    for i in range(4):
        y = 140 - i * 98
        add_rect(draw, 18, y, 14, 8, "#7b3334", "#7b3334")
        add_rect(draw, 74, y, 14, 8, "#7b3334", "#7b3334")


def generate_svg() -> str:
    svg_header = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">\n'
        f'  <rect width="{WIDTH}" height="{HEIGHT}" fill="#bcd8ee"/>\n'
    )
    svg_body = "\n".join(svg_elements)
    svg_footer = "\n</svg>\n"
    return svg_header + svg_body + svg_footer


def main():
    svg_elements.clear()

    # 1) 直接创建完整尺寸 PNG 画布，避免截屏裁切
    image = Image.new("RGB", (WIDTH, HEIGHT), "#bcd8ee")
    draw = ImageDraw.Draw(image)

    # 2) 绘制建筑
    draw_scene_base(draw)
    draw_left_main_tower(draw)
    draw_yellow_shaft(draw)
    draw_right_side_tower(draw)
    draw_roof_frame(draw)
    draw_connector_beams(draw)
    draw_foreground_trees(draw)

    file_dir = Path(__file__).parent

    # 3) 保存完整 PNG（重试版本）
    png_path = file_dir / "building_retry.png"
    image.save(png_path, format="PNG")

    # 4) 同步导出 SVG（重试版本）
    svg_path = file_dir / "building_retry.svg"
    svg_path.write_text(generate_svg(), encoding="utf-8")

    print(f"✓ PNG 已保存: {png_path}")
    print(f"✓ SVG 已保存: {svg_path}")
    print("✓ 输出为完整画布，不会裁切（重试版）")


if __name__ == "__main__":
    main()

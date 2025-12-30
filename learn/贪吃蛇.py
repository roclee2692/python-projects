# -*- coding: utf-8 -*-
import pygame
import random

# ---------- 基本设置 ----------
# 获取屏幕分辨率
pygame.init()
info = pygame.display.Info()
SCREEN_WIDTH, SCREEN_HEIGHT = info.current_w, info.current_h

# 初始窗口大小（可自适应，最大80%屏幕）
WIDTH, HEIGHT = min(1200, int(SCREEN_WIDTH * 0.8)), min(800, int(SCREEN_HEIGHT * 0.8))
BLOCK          = 20          # 蛇身和食物的像素尺寸
INIT_SPEED     = 8           # 初始帧率（越大越快）

# 创建可调节大小的窗口
pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("贪吃蛇（优化版）- 按 F 全屏 | 按 ESC 退出全屏")
screen = pygame.display.get_surface()
clock  = pygame.time.Clock()

# 全屏状态标志
is_fullscreen = False

# ---------- 颜色 ----------
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
GREEN = (  0, 255,   0)
LIGHT_GREEN = (100, 200, 100)
RED   = (255,   0,   0)
ORANGE = (255, 165,   0)
YELLOW = (255, 255,   0)

# ---------- 字体 ----------
# 尝试使用支持中文的字体，如果不可用则使用备选方案
def get_font(size: int):
    """获取支持中文的字体"""
    # Windows上常见的中文字体
    fonts = [
        "Microsoft YaHei",      # 微软雅黑（最推荐）
        "SimHei",               # 黑体
        "FangSong",             # 仿宋
        "KaiTi",                # 楷体
        "Arial Unicode MS",     # Arial Unicode
        "WenQuanYi Zen Hei",    # 文泉驿（Linux）
    ]

    for font_name in fonts:
        try:
            return pygame.font.SysFont(font_name, size)
        except:
            pass

    # 如果都找不到，使用默认字体
    return pygame.font.Font(None, size)

font_small  = get_font(28)
font_medium = get_font(42)
font_large  = get_font(60)

# ---------- 全屏相关函数 ----------
def toggle_fullscreen():
    """切换全屏/窗口模式"""
    global screen, is_fullscreen, WIDTH, HEIGHT

    if is_fullscreen:
        # 退出全屏，恢复窗口模式
        WIDTH, HEIGHT = min(1200, int(SCREEN_WIDTH * 0.8)), min(800, int(SCREEN_HEIGHT * 0.8))
        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        is_fullscreen = False
        pygame.display.set_caption("贪吃蛇（优化版）- 按 F 全屏 | 按 ESC 退出全屏")
    else:
        # 进入全屏
        WIDTH, HEIGHT = SCREEN_WIDTH, SCREEN_HEIGHT
        screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
        is_fullscreen = True
        pygame.display.set_caption("贪吃蛇（优化版）- 按 ESC 退出全屏")

def handle_video_resize(new_width: int, new_height: int):
    """处理窗口大小变化"""
    global screen, WIDTH, HEIGHT, BLOCK
    WIDTH, HEIGHT = new_width, new_height
    screen = pygame.display.get_surface()

# ---------- 工具函数 ----------
def draw_score(score: int, high_score: int = 0) -> None:
    """绘制得分和最高分"""
    img1 = font_small.render(f"得分: {score}", True, WHITE)
    screen.blit(img1, (10, 10))
    if high_score > 0:
        img2 = font_small.render(f"最高: {high_score}", True, YELLOW)
        screen.blit(img2, (10, 45))

def draw_snake(body) -> None:
    """绘制蛇身（蛇头用不同颜色，蛇身用渐变效果）"""
    for i, (x, y) in enumerate(body):
        if i == len(body) - 1:  # 蛇头
            pygame.draw.rect(screen, RED, (x, y, BLOCK, BLOCK))
            pygame.draw.rect(screen, YELLOW, (x, y, BLOCK, BLOCK), 2)
        else:  # 蛇身
            color = GREEN if i % 2 == 0 else LIGHT_GREEN
            pygame.draw.rect(screen, color, (x, y, BLOCK, BLOCK))

def random_food(snake_body) -> tuple[int, int]:
    """生成与网格对齐的随机坐标（不与蛇身重叠）"""
    while True:
        pos = (
            random.randrange(0, WIDTH  // BLOCK) * BLOCK,
            random.randrange(0, HEIGHT // BLOCK) * BLOCK
        )
        if pos not in snake_body:
            return pos

def wait_key(msg: str, sub_msg: str = "") -> str:
    """
    显示提示信息并阻塞等待键盘事件，
    返回按下的有效键 ('c'|'q'|'space'|'1'|'2'|'3')
    """
    screen.fill(BLACK)
    txt1 = font_medium.render(msg, True, RED)
    rect1 = txt1.get_rect(center=(WIDTH // 2, HEIGHT // 3))
    screen.blit(txt1, rect1)

    if sub_msg:
        txt2 = font_small.render(sub_msg, True, WHITE)
        rect2 = txt2.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(txt2, rect2)

    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); quit()
            if event.type == pygame.KEYDOWN:
                # 支持更多按键
                if event.key in (pygame.K_c, pygame.K_q, pygame.K_SPACE, pygame.K_1, pygame.K_2, pygame.K_3):
                    return pygame.key.name(event.key)

def show_difficulty_menu() -> int:
    """显示难度选择菜单，返回初始速度"""
    screen.fill(BLACK)
    title = font_large.render("选择难度", True, YELLOW)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, max(40, HEIGHT // 8)))

    options = [
        ("1 简单 (速度 6)", 6),
        ("2 普通 (速度 8)", 8),
        ("3 困难 (速度 10)", 10),
    ]

    y_pos = max(150, HEIGHT // 4)
    for text, _ in options:
        opt = font_medium.render(text, True, WHITE)
        screen.blit(opt, (WIDTH // 2 - opt.get_width() // 2, y_pos))
        y_pos += max(60, HEIGHT // 6)

    tip = font_small.render("按数字键选择难度", True, LIGHT_GREEN)
    screen.blit(tip, (WIDTH // 2 - tip.get_width() // 2, HEIGHT - 100))

    fullscreen_tip = font_small.render("提示: 按 F 键全屏 | 按 ESC 退出全屏", True, ORANGE)
    screen.blit(fullscreen_tip, (WIDTH // 2 - fullscreen_tip.get_width() // 2, HEIGHT - 50))

    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); quit()
            if event.type == pygame.VIDEORESIZE:
                handle_video_resize(event.w, event.h)
                screen.fill(BLACK)
                title = font_large.render("选择难度", True, YELLOW)
                screen.blit(title, (WIDTH // 2 - title.get_width() // 2, max(40, HEIGHT // 8)))
                y_pos = max(150, HEIGHT // 4)
                for text, _ in options:
                    opt = font_medium.render(text, True, WHITE)
                    screen.blit(opt, (WIDTH // 2 - opt.get_width() // 2, y_pos))
                    y_pos += max(60, HEIGHT // 6)
                tip = font_small.render("按数字键选择难度", True, LIGHT_GREEN)
                screen.blit(tip, (WIDTH // 2 - tip.get_width() // 2, HEIGHT - 100))
                fullscreen_tip = font_small.render("提示: 按 F 键全屏 | 按 ESC 退出全屏", True, ORANGE)
                screen.blit(fullscreen_tip, (WIDTH // 2 - fullscreen_tip.get_width() // 2, HEIGHT - 50))
                pygame.display.flip()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return 6
                elif event.key == pygame.K_2:
                    return 8
                elif event.key == pygame.K_3:
                    return 10

# ---------- 主逻辑 ----------
def main():
    # 显示难度选择菜单
    speed          = show_difficulty_menu()
    base_speed     = speed

    direction      = pygame.K_RIGHT           # 当前朝向
    next_direction = pygame.K_RIGHT           # 下一帧要执行的朝向（缓存机制）
    snake          = [(WIDTH // 2, HEIGHT // 2)]
    food           = random_food(snake)
    grow_pending   = 0                        # 吃到食物后要增长的节数
    running, pause = True, False
    frame_count    = 0                        # 用于控制移动频率

    while running:
        # -------- 事件处理 --------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # 窗口大小变化事件
            if event.type == pygame.VIDEORESIZE:
                handle_video_resize(event.w, event.h)

            if event.type == pygame.KEYDOWN:
                key = event.key
                # F 键切换全屏
                if key == pygame.K_f:
                    toggle_fullscreen()
                    continue
                # ESC 键退出全屏
                if key == pygame.K_ESCAPE and is_fullscreen:
                    toggle_fullscreen()
                    continue

                # WSAD + 方向键皆可，更新next_direction而不是直接改direction
                # 这样可以在等待下一帧时执行操作，提高反应速度
                if key in (pygame.K_UP, pygame.K_w)    and direction not in (pygame.K_DOWN,  pygame.K_s):
                    next_direction = pygame.K_UP
                elif key in (pygame.K_DOWN, pygame.K_s) and direction not in (pygame.K_UP,    pygame.K_w):
                    next_direction = pygame.K_DOWN
                elif key in (pygame.K_LEFT, pygame.K_a) and direction not in (pygame.K_RIGHT, pygame.K_d):
                    next_direction = pygame.K_LEFT
                elif key in (pygame.K_RIGHT, pygame.K_d) and direction not in (pygame.K_LEFT,  pygame.K_a):
                    next_direction = pygame.K_RIGHT
                elif key == pygame.K_SPACE:   # 暂停/恢复
                    pause = not pause

        if pause:
            wait_key("游戏暂停", "空格继续")
            pause = False
            continue

        # -------- 逻辑更新 --------
        # 只在足够的帧数后才移动蛇（基于速度）
        frame_count += 1
        move_interval = int(30 / speed)  # 根据速度动态计算移动间隔

        if frame_count >= move_interval:
            frame_count = 0
            direction = next_direction  # 应用缓存的方向

            x, y = snake[-1]
            if   direction in (pygame.K_UP,    pygame.K_w): y -= BLOCK
            elif direction in (pygame.K_DOWN,  pygame.K_s): y += BLOCK
            elif direction in (pygame.K_LEFT,  pygame.K_a): x -= BLOCK
            elif direction in (pygame.K_RIGHT, pygame.K_d): x += BLOCK

            new_head = (x % WIDTH, y % HEIGHT)  # 越界穿墙
            if new_head in snake:               # 撞到自己
                key = wait_key("游戏结束！", "C重新开始  Q退出")
                if key == 'c':
                    return main()
                else:
                    break

            snake.append(new_head)
            if new_head == food:
                grow_pending += 1
                food = random_food(snake)  # 重新生成食物时检查与蛇的碰撞
                # 每吃10个食物加速一次（最高速度30）
                if len(snake) % 10 == 0:
                    speed = min(speed + 0.5, 15)
            if grow_pending:
                grow_pending -= 1
            else:
                snake.pop(0)

        # -------- 绘制 --------
        screen.fill(BLACK)
        pygame.draw.rect(screen, ORANGE, (*food, BLOCK, BLOCK))
        # 食物添加脉冲效果
        frame_mod = (frame_count % 10)
        if frame_mod > 5:
            pygame.draw.rect(screen, YELLOW, (*food, BLOCK, BLOCK), 2)

        draw_snake(snake)
        draw_score(len(snake) - 1)

        # 显示当前速度和难度
        speed_text = font_small.render(f"速度: {speed:.1f}", True, LIGHT_GREEN)
        screen.blit(speed_text, (WIDTH - 200, 10))

        # 显示快捷键提示
        shortcut_text = font_small.render("F-全屏  ESC-退出  SPACE-暂停", True, WHITE)
        screen.blit(shortcut_text, (10, HEIGHT - 35))

        pygame.display.flip()
        clock.tick(30)  # 固定60帧，通过move_interval控制蛇的实际移动速度

    pygame.quit()

if __name__ == "__main__":
    main()
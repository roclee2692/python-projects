# -*- coding: utf-8 -*-
import pygame
import random

# ---------- 基本设置 ----------
WIDTH, HEIGHT = 800, 600
BLOCK          = 20          # 蛇身和食物的像素尺寸
INIT_SPEED     = 8           # 初始帧率（越大越快）

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("贪吃蛇（优化版）")
clock  = pygame.time.Clock()

# ---------- 颜色 ----------
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
GREEN = (  0, 255,   0)
RED   = (255,   0,   0)

# ---------- 字体 ----------
font_small  = pygame.font.SysFont("bahnschrift", 28)
font_medium = pygame.font.SysFont("comicsansms", 42)

# ---------- 工具函数 ----------
def draw_score(score: int) -> None:
    img = font_small.render(f"得分: {score}", True, WHITE)
    screen.blit(img, (10, 10))

def draw_snake(body) -> None:
    for x, y in body:
        pygame.draw.rect(screen, GREEN, (x, y, BLOCK, BLOCK))

def random_food() -> tuple[int, int]:
    """生成与网格对齐的随机坐标"""
    return (
        random.randrange(0, WIDTH  // BLOCK) * BLOCK,
        random.randrange(0, HEIGHT // BLOCK) * BLOCK
    )

def wait_key(msg: str, sub_msg: str = "") -> str:
    """
    显示提示信息并阻塞等待键盘事件，
    返回按下的有效键 ('c'|'q'|'space')
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
                if event.key in (pygame.K_c, pygame.K_q, pygame.K_SPACE):
                    return pygame.key.name(event.key)

# ---------- 主逻辑 ----------
def main():
    speed          = INIT_SPEED               # 当前帧率
    direction      = pygame.K_RIGHT           # 初始朝向 →    
    snake          = [(WIDTH // 2, HEIGHT // 2)]
    food           = random_food()
    grow_pending   = 0                        # 吃到食物后要增长的节数
    running, pause = True, False

    while running:
        # -------- 事件处理 --------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                key = event.key
                # WSAD + 方向键皆可
                if key in (pygame.K_UP, pygame.K_w)    and direction not in (pygame.K_DOWN,  pygame.K_s):
                    direction = pygame.K_UP
                elif key in (pygame.K_DOWN, pygame.K_s) and direction not in (pygame.K_UP,    pygame.K_w):
                    direction = pygame.K_DOWN
                elif key in (pygame.K_LEFT, pygame.K_a) and direction not in (pygame.K_RIGHT, pygame.K_d):
                    direction = pygame.K_LEFT
                elif key in (pygame.K_RIGHT, pygame.K_d) and direction not in (pygame.K_LEFT,  pygame.K_a):
                    direction = pygame.K_RIGHT
                elif key == pygame.K_SPACE:   # 暂停/恢复
                    pause = not pause

        if pause:
            wait_key("游戏暂停", "空格继续")
            pause = False
            continue

        # -------- 逻辑更新 --------
        x, y = snake[-1]
        if   direction in (pygame.K_UP,    pygame.K_w): y -= BLOCK
        elif direction in (pygame.K_DOWN,  pygame.K_s): y += BLOCK
        elif direction in (pygame.K_LEFT,  pygame.K_a): x -= BLOCK
        elif direction in (pygame.K_RIGHT, pygame.K_d): x += BLOCK

        new_head = (x % WIDTH, y % HEIGHT)  # 越界穿墙；如不想穿墙可删去 % 并检测撞墙
        if new_head in snake:               # 撞到自己
            key = wait_key("游戏结束！", "C重新开始  Q退出")
            if key == 'c':
                return main()
            else:
                break

        snake.append(new_head)
        if new_head == food:
            grow_pending += 1
            food = random_food()
            speed = min(speed + 0.2, 30)    # 随得分略微加速
        if grow_pending:
            grow_pending -= 1
        else:
            snake.pop(0)

        # -------- 绘制 --------
        screen.fill(BLACK)
        pygame.draw.rect(screen, RED, (*food, BLOCK, BLOCK))
        draw_snake(snake)
        draw_score(len(snake) - 1)
        pygame.display.flip()
        clock.tick(speed)

    pygame.quit()

if __name__ == "__main__":
    main()
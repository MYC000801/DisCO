import random
import collections
import heapq
from collections import deque
from typing import Tuple, List


def generate_branchy_maze(
    n: int,
    branchiness: float = 0.8,
    farthest_goal: bool = True
) -> Tuple[Tuple[int, int], Tuple[int, int], List[List[int]]]:
    """
    随机生成一张“分叉可调”的 perfect maze，并随机选取 start / goal。

    参数
    ----
    n            : 迷宫边长 (方形，单位格)
    branchiness  : 0‑1，越大分叉越多；≈0 -> DFS，≈1 -> Prim
    farthest_goal: True -> 选离 start 最远的格子当 goal；False -> 随机可达格

    返回
    ----
    (start, goal, maze)   其中 maze[i][j] == 0 表路, 1 表墙
    """
    # ---------- 初始化 ----------
    maze = [[1] * n for _ in range(n)]

    # start 随机挑一个偶数坐标 (保证格子间隔 2 时相邻仍在网格内)
    def rand_even(limit):                       # 0,2,4,… < limit
        max_even = limit - 1 if limit % 2 else limit - 2
        return random.randrange(0, max_even + 1, 2)

    start = (rand_even(n), rand_even(n))

    # ---------- Growing‑Tree 主循环 ----------
    def neighbors(x, y):
        # 返回: (邻居 x, 邻居 y, (wx, wy) 墙坐标增量)
        for dx, dy in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n:
                yield nx, ny, dx // 2, dy // 2

    maze[start[0]][start[1]] = 0
    active = [start]

    while active:
        idx = -1 if random.random() > branchiness else random.randrange(len(active))
        x, y = active[idx]

        unvisited = [(nx, ny, wx, wy) for nx, ny, wx, wy in neighbors(x, y)
                     if maze[nx][ny] == 1]

        if unvisited:
            nx, ny, wx, wy = random.choice(unvisited)
            maze[x + wx][y + wy] = 0         # 打通墙
            maze[nx][ny] = 0
            active.append((nx, ny))
        else:
            active.pop(idx)                  # 死胡同：移除

    # ---------- 选取 goal ----------
    def bfs_farthest(src):
        """BFS 找到离 src 最远的可通行格；返回坐标"""
        vis = {src}
        q = deque([(src[0], src[1], 0)])
        far, far_dist = src, 0
        while q:
            x, y, d = q.popleft()
            if d > far_dist:
                far, far_dist = (x, y), d
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and maze[nx][ny] == 0 and (nx, ny) not in vis:
                    vis.add((nx, ny))
                    q.append((nx, ny, d + 1))
        return far

    if farthest_goal:
        goal = bfs_farthest(start)
    else:
        path_cells = [(i, j) for i in range(n) for j in range(n)
                      if maze[i][j] == 0 and (i, j) != start]
        goal = random.choice(path_cells)

    # 保证 start / goal 两格都是路
    maze[start[0]][start[1]] = 0
    maze[goal[0]][goal[1]]   = 0

    return start, goal, maze


# ────────────────────────── 2. 工具函数 ──────────────────────────
DIRS = [ (0, -1), (0, 1), (-1, 0), (1, 0) ]          # 左、右、上、下
def to1(p):                                           # 0‑based → 字符串 "(x, y)" 1‑based
    return f"({p[0] + 1}, {p[1] + 1})"

def observation_str(pos, maze, goal):
    n, (x, y) = len(maze), pos
    parts = []
    for dx, dy in DIRS:                               # 左→右→上→下
        nx, ny = x + dx, y + dy
        if 0 <= nx < n and 0 <= ny < n:
            if (nx, ny) == goal:
                state = "exit"
            else:
                state = "path" if maze[nx][ny] == 0 else "wall"
        else:
            state = "wall"
        parts.append(f"({nx + 1}, {ny + 1}): {state}")
    return "; ".join(parts)

# 已知网格内的 BFS 最短路（返回 deque，空表示不可达或就在原地）
def shortest_path(src, dst, walkable):
    if src == dst:
        return collections.deque()
    q = collections.deque([src])
    parent = {src: None}
    while q:
        cur = q.popleft()
        if cur == dst:
            break
        for dx, dy in DIRS:
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt in walkable and nxt not in parent:
                parent[nxt] = cur
                q.append(nxt)
    if dst not in parent:
        return collections.deque()
    path = collections.deque()
    cur = dst
    while cur != src:
        path.appendleft(cur)
        cur = parent[cur]
    return path
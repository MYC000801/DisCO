import random
import collections
from collections import deque
from typing import Tuple, List
import pandas as pd
from datasets import Dataset
from tqdm import trange
from tqdm import tqdm
import argparse
import os
import json



parser = argparse.ArgumentParser(description="Generate multi-trajectory maze SFT dataset.")
parser.add_argument("--local_path", type=str, default='./data/maze', help="Path to save the generated parquet file.")
parser.add_argument("--num_samples", type=int, default=100000, help="Number of maze samples to generate.")
parser.add_argument("--maze_size", type=int, default=7, help="Size of the maze (NxN).")
parser.add_argument("--step_per_traj", type=int, default=15, help="Max steps per trajectory.")
parser.add_argument("--traj_per_chat", type=int, default=5, help="Number of trajectories per chat.")

args = parser.parse_args()

local_dir = args.local_path
NUM_SAMPLES = args.num_samples
MAZE_SIZE = args.maze_size
STEP_PER_TRAJ = args.step_per_traj
TRAJ_PER_CHAT = args.traj_per_chat



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


def build_multi_trajectory_chat_history(maze, start, goal, K=3, N=10):
    """
    构造包含 K 个 trajectory 的 chat history，每个 trajectory 最多 N 步
    agent 可以利用前面所有 trajectory 收集到的信息
    """
    DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]     # 左右上下

    def to1(p):
        return f"({p[0] + 1}, {p[1] + 1})"

    def obs_content(pos, traj_id):
        n, (x, y) = len(maze), pos
        parts = []
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n:
                state = (
                    "exit"  if (nx, ny) == goal else
                    "path"  if maze[nx][ny] == 0 else
                    "wall"
                )
            else:
                state = "wall"
            parts.append(f"({nx + 1}, {ny + 1}): {state}")
        obs_str = ", ".join(parts)
        return f"Trajectory {traj_id}: {obs_str}"

    n = len(maze)
    # Global knowledge that persists across all trajectories
    global_map = [[None for _ in range(n)] for _ in range(n)]
    global_map[start[0]][start[1]] = 0
    
    # Track all positions visited across all trajectories
    global_visited = set()
    
    # Track exploration frontiers (positions adjacent to unknown areas)
    exploration_frontiers = set()
    
    # Initialize messages with system prompt
    messages = [{"role": "system", "content": "You are an intelligent agent navigating a maze across multiple attempts. At each step, you receive an observation with trajectory ID and four adjacent cells (coordinates + 'path'/'wall'/'exit'). Learn from previous trajectories to navigate more efficiently. Choose exactly one adjacent 'path' or 'exit' cell to move into. Output your next move as coordinates (row, col) only."}]

    def update_global_map(pos):
        """Update global map with observations from current position"""
        for dx, dy in DIRS:
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < n and 0 <= ny < n:
                if global_map[nx][ny] is None:
                    global_map[nx][ny] = maze[nx][ny]
                    # If it's a path, add it to exploration consideration
                    if maze[nx][ny] == 0:
                        exploration_frontiers.add((nx, ny))

    def get_known_walkable():
        """Get all currently known walkable positions"""
        return set(
            (i, j) for i in range(n) for j in range(n)
            if global_map[i][j] == 0 or (i, j) == goal
        )

    def get_exploration_targets(pos, known_walkable):
        """Get positions that are good for exploration (adjacent to unknown areas)"""
        targets = []
        for i in range(n):
            for j in range(n):
                if (i, j) in known_walkable:
                    # Check if this position has unknown neighbors
                    has_unknown_neighbor = False
                    for dx, dy in DIRS:
                        nx, ny = i + dx, j + dy
                        if 0 <= nx < n and 0 <= ny < n and global_map[nx][ny] is None:
                            has_unknown_neighbor = True
                            break
                    if has_unknown_neighbor and is_reachable(pos, (i, j), known_walkable):
                        targets.append((i, j))
        return targets

    def is_reachable(src, dst, walkable):
        if src == dst:
            return True
        q = collections.deque([src])
        visited = {src}
        while q:
            cur = q.popleft()
            if cur == dst:
                return True
            for dx, dy in DIRS:
                nxt = (cur[0] + dx, cur[1] + dy)
                if nxt in walkable and nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        return False

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

    def smart_next_move(pos, local_visited, known_walkable):
        """
        Smart strategy for choosing next move:
        1. Go directly to goal if known and reachable
        2. Explore unvisited known areas (prioritize closer to goal)
        3. Explore frontier areas (positions adjacent to unknown regions)
        4. Fall back to any reachable position
        """
        
        # Strategy 1: Direct path to goal if known and reachable
        if goal in known_walkable and goal not in local_visited:
            if is_reachable(pos, goal, known_walkable):
                path = shortest_path(pos, goal, known_walkable)
                if path:
                    return path.popleft()
        
        # Strategy 2: Visit unvisited known areas (prioritize by distance to goal)
        unvisited_known = [
            p for p in known_walkable 
            if p not in local_visited and p != pos and p not in global_visited
        ]
        if unvisited_known:
            # Sort by distance to goal
            target = min(unvisited_known, key=lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1]))
            if is_reachable(pos, target, known_walkable):
                path = shortest_path(pos, target, known_walkable)
                if path:
                    return path.popleft()
        
        # Strategy 3: Explore areas that might reveal new information
        exploration_targets = get_exploration_targets(pos, known_walkable)
        if exploration_targets:
            # Prioritize targets closer to goal
            target = min(exploration_targets, 
                        key=lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1]))
            if target not in local_visited:
                path = shortest_path(pos, target, known_walkable)
                if path:
                    return path.popleft()
        
        # Strategy 4: Visit any unvisited known position in this trajectory
        any_unvisited = [
            p for p in known_walkable 
            if p not in local_visited and p != pos
        ]
        if any_unvisited:
            target = min(any_unvisited, key=lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1]))
            if is_reachable(pos, target, known_walkable):
                path = shortest_path(pos, target, known_walkable)
                if path:
                    return path.popleft()
                    
        return None

    for k in range(1, K + 1):
        pos = start
        trajectory_success = False
        local_visited = set()
        local_visited.add(pos)

        if messages[-1]["role"] == "user":
            messages[-1]["content"] = messages[-1]["content"] + "\n\n" + obs_content(pos, k)
        else:
            messages.append({"role": "user", "content": obs_content(pos, k)})
        update_global_map(pos)

        for step in range(N):
            update_global_map(pos)
            
            if pos == goal:
                trajectory_success = True
                break

            known_walkable = get_known_walkable()
            
            # Use smart strategy to choose next move
            next_pos = smart_next_move(pos, local_visited, known_walkable)
            
            if next_pos is None:
                # No valid moves available
                break
                
            # Execute the move
            messages.append({"role": "assistant", "content": to1(next_pos)})
            pos = next_pos
            local_visited.add(pos)
            global_visited.add(pos)  # Track globally visited positions
            
            # Continue with observation if not at goal and not last step
            if step < N - 1 and pos != goal:
                messages.append({"role": "user", "content": obs_content(pos, k)})

        # Add trajectory end message
        if trajectory_success:
            messages.append({"role": "user", "content": "Arrive the goal! Let's try again."})
        else:
            messages.append({"role": "user", "content": "You did not find the Exit. Let's try again."})

    return messages


def generate_maze_rl_dataset(num_samples=1000, maze_size=7, data_source="maze_navigation", steps=20, episodes=5):
    """
    生成迷宫导航数据集，格式符合VERL训练要求
    """
    
    def observation_to_question(pos, maze, goal):
        """将当前位置的观测转换为问题描述"""
        DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 左右上下
        n, (x, y) = len(maze), pos
        parts = []
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n:
                state = (
                    "exit" if (nx, ny) == goal else
                    "path" if maze[nx][ny] == 0 else
                    "wall"
                )
            else:
                state = "wall"
            parts.append(f"({nx + 1}, {ny + 1}): {state}")
        
        return f"{', '.join(parts)}"
    
    def maze_to_solution(maze, start, goal):
        """将迷宫转换为解决方案字符串（包含迷宫布局和起点终点信息）"""
        solution_data = {
            "maze": maze,
            "start": start,
            "goal": goal,
            "size": len(maze)
        }
        return str(solution_data)
    
    def make_map_fn(split):
        def process_fn(example, idx):
            # 生成迷宫
            start, goal, maze = generate_branchy_maze(maze_size)
            
            # 生成起点观测作为问题
            question_raw = observation_to_question(start, maze, goal)
            question = "Trajectory 1: " + question_raw 
            
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": "You are an intelligent agent navigating a maze across multiple attempts. At each step, you receive an observation with trajectory ID and four adjacent cells (coordinates + 'path'/'wall'/'exit'). Learn from previous trajectories to navigate more efficiently. Choose exactly one adjacent 'path' or 'exit' cell to move into. Output your next move as coordinates (row, col) only.",
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "ability": "navigation",  # 改为导航能力
                "reward_model": {"style": "rule", "ground_truth": goal},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "maze": maze,
                    "start": start,
                    "goal": goal,
                    "steps": steps,
                    "episodes": episodes,
                    "interaction_kwargs": {
                        "query": question,
                        "maze": maze,
                        "start": start,
                        "goal": goal,
                        "steps": steps,
                        "episodes": episodes,
                    },
                },
            }
            return data
        return process_fn
    
    # 生成数据集
    dataset = []
    process_fn = make_map_fn("train")
    
    avg_arrive = 0
    count = 0
    for i in tqdm(range(num_samples), desc="Generating maze samples"):
        example = {}  # 空的example，因为我们直接在process_fn中生成迷宫
        data_item = process_fn(example, i)
        chat_history = build_multi_trajectory_chat_history(maze=data_item["extra_info"]["maze"],
                                           start=data_item["extra_info"]["start"],
                                           goal=data_item["extra_info"]["goal"],
                                           K=data_item["extra_info"]["episodes"],
                                           N=data_item["extra_info"]["steps"])
        arrive_numv = sum([1 for msg in chat_history if  "Arrive the goal! Let's try again." in msg['content']])

        # TODO: Control the difficulty of the maze
        if arrive_numv > 0:
            dataset.append(data_item)
            avg_arrive += arrive_numv
            count += 1

    print(f"Average arrive rate for SFT model: {avg_arrive / count}")

    return dataset



os.makedirs(local_dir, exist_ok=True)

print("Generating maze datasets...")



# 生成训练集 (10000 samples)
print("Generating training set...")
train_dataset_raw = generate_maze_rl_dataset(num_samples=NUM_SAMPLES, maze_size=MAZE_SIZE, data_source="maze_navigation", steps=STEP_PER_TRAJ, episodes=TRAJ_PER_CHAT)

# 生成测试集 (500 samples)
print("Generating test set...")
test_dataset_raw = generate_maze_rl_dataset(num_samples=500, maze_size=MAZE_SIZE, data_source="maze_navigation", steps=STEP_PER_TRAJ, episodes=TRAJ_PER_CHAT)

# 更新split标记
for item in train_dataset_raw:
    item['extra_info']['split'] = 'train'

for item in test_dataset_raw:
    item['extra_info']['split'] = 'test'

print(f"Generated {len(train_dataset_raw)} training samples and {len(test_dataset_raw)} test samples")

# 转换为HuggingFace Dataset格式
train_df = pd.DataFrame(train_dataset_raw)
test_df = pd.DataFrame(test_dataset_raw)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 保存为parquet格式（按照GSM8K格式）
train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

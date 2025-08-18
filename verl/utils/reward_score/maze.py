# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

def get_observation_chat(pos, maze_grid):
    """
    生成 observation 格式的 chat，返回 [{'role': 'user', 'content': ...}]
    
    Args:
        pos: 当前位置 (x, y)，1-based 坐标
        maze_grid: 迷宫格栅，0-based 索引
    """
    DIRS = [ (0, -1), (0, 1), (-1, 0), (1, 0) ]  # 左、右、上、下
    n = len(maze_grid)
    x, y = pos
    parts = []
    for dx, dy in DIRS:
        nx, ny = x + dx, y + dy
        # 检查是否在迷宫边界内
        if 1 <= nx <= n and 1 <= ny <= n:
            # 转换为0-based索引访问迷宫
            if maze_grid[nx-1][ny-1] == -1:
                state = "exit"
            else:
                state = "path" if maze_grid[nx-1][ny-1] == 0 else "wall"
        else:
            state = "wall"
        parts.append(f"({nx}, {ny}): {state}")
    obs_str = ", ".join(parts)
    return [{'role': 'user', 'content': obs_str}]

def get_valid_moves(user_content: str):
    """
    从用户观察内容中提取有效移动位置
    
    Args:
        user_content: 观察字符串，格式如 '(5, 4): path, (5, 6): wall, (4, 5): path, (6, 5): path'
    
    Returns:
        有效移动位置的坐标字符串列表，如 ['(5, 4)', '(4, 5)', '(6, 5)']
    """
    valid_moves = []
    
    # 使用正则表达式匹配所有的坐标和状态对
    pattern = r'\((\d+),\s*(\d+)\):\s*(path|wall|exit)'
    matches = re.findall(pattern, user_content)
    
    for x, y, state in matches:
        if state in ['path', 'exit']:  # path 和 exit 都是有效移动
            valid_moves.append(f"({x}, {y})")
    
    return valid_moves

# 获取模型输出的坐标字符串，并提取两个数字
def extract_move(solution_str):
    # 匹配 (多位数字, 多位数字) 格式
    match = re.search(r"\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)", solution_str)
    if match:
        coord_str = match.group(0)
        row = int(match.group(1))
        col = int(match.group(2))
        return coord_str, row, col
    return None, None, None





def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for Maze.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_move(solution_str=solution_str)
    valid_moves = get_valid_moves(ground_truth)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score

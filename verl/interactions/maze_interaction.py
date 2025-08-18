# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import logging
import os
import re
import ast
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from verl.utils.reward_score import maze_mt

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))



class MazeInteraction(BaseInteraction):
    """A demo interaction for calculating the reward of maze navigation.

    - `start_interaction`: start a interaction instance for a trajectory.
    - `generate_response`: generate the response of the user.
    - `calculate_score`: calculate the score of the interaction.
    - `finalize_interaction`: finalize the interaction instance.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}

    def _parse_maze_string(self, maze_str: str) -> List[List[int]]:
        """将字符串格式的迷宫转换为二维表格
        
        Args:
            maze_str: 字符串格式的迷宫，可能是以下格式之一：
                - JSON字符串: "[[0,1,0],[1,0,1],[0,0,-1]]"
                - Python列表字符串: "[[0,1,0],[1,0,1],[0,0,-1]]"
                
        Returns:
            二维列表表示的迷宫
        """
        try:
            # 尝试使用 ast.literal_eval 解析
            maze_grid = ast.literal_eval(maze_str)
            if isinstance(maze_grid, list) and all(isinstance(row, list) for row in maze_grid):
                return maze_grid
        except (ValueError, SyntaxError):
            pass
        
        try:
            # 如果 ast.literal_eval 失败，尝试 eval（不推荐，但作为备选）
            maze_grid = eval(maze_str)
            if isinstance(maze_grid, list) and all(isinstance(row, list) for row in maze_grid):
                return maze_grid
        except:
            pass
        
        # 如果都失败了，抛出异常
        raise ValueError(f"无法解析迷宫字符串: {maze_str}")

    def _parse_coordinate_string(self, coord_str: str) -> Tuple[int, int]:
        """将字符串格式的坐标转换为元组
        
        Args:
            coord_str: 字符串格式的坐标，如 "(1, 2)" 或 "[1, 2]"
            
        Returns:
            坐标元组 (x, y)
        """
        try:
            # 尝试使用 ast.literal_eval 解析
            coord = ast.literal_eval(coord_str)
            if isinstance(coord, (list, tuple)) and len(coord) == 2:
                return tuple(coord)
        except (ValueError, SyntaxError):
            pass
        
        # 如果失败，尝试正则表达式提取
        match = re.search(r'[\[\(](\d+),\s*(\d+)[\]\)]', coord_str)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        
        raise ValueError(f"无法解析坐标字符串: {coord_str}")

    async def start_interaction(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, maze: Optional[Union[str, List[List[int]]]] = None, start: Optional[Union[str, Tuple[int, int]]] = None, goal: Optional[Union[str, Tuple[int, int]]] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        
        # 将字符串参数转换为适当的数据类型
        maze_grid = self._parse_maze_string(maze) if isinstance(maze, str) else maze
        start_pos = self._parse_coordinate_string(start) if isinstance(start, str) else start
        goal_pos = self._parse_coordinate_string(goal) if isinstance(goal, str) else goal
        
        maze_grid[goal_pos[0]][goal_pos[1]] = -1  # 确保目标位置标记为-1    

        self._instance_dict[instance_id] = {
            "response": "",
            "user_content": "",
            "step_count": 0,
            "episode_id": 1,
            "maze": maze_grid,
            "goal": goal_pos,
            "start": start_pos,
            "reward": 0.0,
        }
        return instance_id

    async def generate_response(self, instance_id: str, messages: List[Dict[str, Any]], **kwargs) -> Tuple[bool, str, float, dict]:
        content = ""
        user_content = ""
        for i in range(len(messages) - 1, -1, -1):
            item = messages[i]
            if item.get("role") == "assistant":
                content = item.get("content")
            elif item.get("role") == "user":
                user_content = item.get("content")
            
            # 如果已经找到了assistant和user的内容，就退出循环
            if content and user_content:
                break

        # 更新实例字典中的response为最新的assistant回复
        self._instance_dict[instance_id]["response"] = content
        self._instance_dict[instance_id]["user_content"] = user_content

        self._instance_dict[instance_id]["step_count"] = sum([f"Trajectory {self._instance_dict[instance_id]['episode_id']}: " in msg["content"] for msg in messages if "content" in msg])
        should_terminate_sequence, response, reward = await self.calculate_feedback(instance_id)
    
        return should_terminate_sequence, response, reward, {}

    async def calculate_feedback(self, instance_id: str, **kwargs):
        valid_moves = maze_mt.get_valid_moves(self._instance_dict[instance_id]["user_content"])
        model_move, x, y = maze_mt.extract_move(self._instance_dict[instance_id]["response"])
        reward = 0.0
        if model_move in valid_moves:
            should_terminate_sequence = False
            pos = (x, y)
            # 检查是否到达目标
            goal = self._instance_dict[instance_id]["goal"]
            # arrive the goal
            if x == goal[0] + 1 and y == goal[1] + 1:
                response = "Arrive the goal! Let's try again.\n\n"
                self._instance_dict[instance_id]["episode_id"] += 1
                pos = self._instance_dict[instance_id]["start"]

                reward = 1.0  # 到达目标奖励
                
                maze_grid = self._instance_dict[instance_id]["maze"]
                response = response + maze_mt.get_observation_chat(pos, maze_grid, self._instance_dict[instance_id]["episode_id"])
            elif self._instance_dict[instance_id]["step_count"] >= 15:
                response = "You did not find the Exit. Let's try again.\n\n"
                self._instance_dict[instance_id]["episode_id"] += 1
                pos = self._instance_dict[instance_id]["start"]
                
                maze_grid = self._instance_dict[instance_id]["maze"]
                response = response + maze_mt.get_observation_chat(pos, maze_grid, self._instance_dict[instance_id]["episode_id"])                
            else:

                maze_grid = self._instance_dict[instance_id]["maze"]
                response = maze_mt.get_observation_chat(pos, maze_grid, self._instance_dict[instance_id]["episode_id"])

        else:
            response = "You did not find the Exit. Let's try again.\n\n"
            self._instance_dict[instance_id]["episode_id"] += 1
            pos = self._instance_dict[instance_id]["start"]
            
            maze_grid = self._instance_dict[instance_id]["maze"]
            response = response + maze_mt.get_observation_chat(pos, maze_grid, self._instance_dict[instance_id]["episode_id"])  

        if self._instance_dict[instance_id]["episode_id"] > 5:
            should_terminate_sequence = True
        else:
            should_terminate_sequence = False

        

        return should_terminate_sequence, response, reward
    

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]

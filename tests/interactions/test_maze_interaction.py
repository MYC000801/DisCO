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

from unittest.mock import patch

import pytest

from verl.interactions.maze_interaction import MazeInteraction


class TestMazeInteraction:
    """Test cases for MazeInteraction class."""

    def setup_method(self):
        """Set up test environment before each test method."""
        self.config = {}
        self.interaction = MazeInteraction(self.config)
        
        # Sample maze data for testing
        self.sample_maze = [
            [0, 1, 0],
            [0, 0, 0],
            [1, 0, -1]
        ]
        self.sample_start = (1, 1)
        self.sample_goal = (3, 3)
        self.sample_maze_str = "[[0, 1, 0], [0, 0, 0], [1, 0, -1]]"
        self.sample_start_str = "(1, 1)"
        self.sample_goal_str = "(3, 3)"

    def test_init(self):
        """Test MazeInteraction initialization."""
        assert self.interaction._instance_dict == {}
        assert self.interaction.config == self.config

    def test_parse_maze_string_valid(self):
        """Test _parse_maze_string with valid string."""
        result = self.interaction._parse_maze_string(self.sample_maze_str)
        assert result == self.sample_maze
        
    def test_parse_maze_string_invalid(self):
        """Test _parse_maze_string with invalid string."""
        with pytest.raises(ValueError):
            self.interaction._parse_maze_string("invalid maze string")

    def test_parse_coordinate_string_valid(self):
        """Test _parse_coordinate_string with valid string."""
        result = self.interaction._parse_coordinate_string(self.sample_start_str)
        assert result == self.sample_start
        
        # Test with square brackets
        result = self.interaction._parse_coordinate_string("[2, 3]")
        assert result == (2, 3)

    def test_parse_coordinate_string_invalid(self):
        """Test _parse_coordinate_string with invalid string."""
        with pytest.raises(ValueError):
            self.interaction._parse_coordinate_string("invalid coordinate")

    @pytest.mark.asyncio
    async def test_start_interaction_with_instance_id_strings(self):
        """Test start_interaction with provided instance_id using string parameters."""
        instance_id = "test_instance"

        result_id = await self.interaction.start_interaction(
            instance_id=instance_id,
            ground_truth="(3, 3)",
            maze=self.sample_maze_str,
            start=self.sample_start_str,
            goal=self.sample_goal_str
        )

        assert result_id == instance_id
        assert instance_id in self.interaction._instance_dict
        assert self.interaction._instance_dict[instance_id]["response"] == ""
        assert self.interaction._instance_dict[instance_id]["user_content"] == ""
        assert self.interaction._instance_dict[instance_id]["ground_truth"] == "(3, 3)"
        assert self.interaction._instance_dict[instance_id]["maze"] == self.sample_maze
        assert self.interaction._instance_dict[instance_id]["start"] == self.sample_start
        assert self.interaction._instance_dict[instance_id]["goal"] == self.sample_goal
        assert self.interaction._instance_dict[instance_id]["reward"] == 0.0

    @pytest.mark.asyncio
    async def test_start_interaction_with_native_types(self):
        """Test start_interaction with native Python types."""
        instance_id = "test_instance"

        result_id = await self.interaction.start_interaction(
            instance_id=instance_id,
            ground_truth="(3, 3)",
            maze=self.sample_maze,
            start=self.sample_start,
            goal=self.sample_goal
        )

        assert result_id == instance_id
        assert self.interaction._instance_dict[instance_id]["maze"] == self.sample_maze
        assert self.interaction._instance_dict[instance_id]["start"] == self.sample_start
        assert self.interaction._instance_dict[instance_id]["goal"] == self.sample_goal

    @pytest.mark.asyncio
    async def test_start_interaction_without_instance_id(self):
        """Test start_interaction without provided instance_id (auto-generated)."""
        result_id = await self.interaction.start_interaction(
            maze=self.sample_maze_str,
            start=self.sample_start_str,
            goal=self.sample_goal_str
        )

        assert result_id is not None
        assert len(result_id) == 36  # UUID4 length
        assert result_id in self.interaction._instance_dict

    @pytest.mark.asyncio
    async def test_start_interaction_without_parameters(self):
        """Test start_interaction without maze parameters."""
        instance_id = "test_instance"

        result_id = await self.interaction.start_interaction(instance_id=instance_id)

        assert result_id == instance_id
        assert self.interaction._instance_dict[instance_id]["maze"] is None
        assert self.interaction._instance_dict[instance_id]["start"] is None
        assert self.interaction._instance_dict[instance_id]["goal"] is None

    @pytest.mark.asyncio
    async def test_generate_response_valid_move_not_goal(self):
        """Test generate_response with valid move that doesn't reach goal."""
        instance_id = "test_instance"
        
        # Setup instance
        await self.interaction.start_interaction(
            instance_id=instance_id,
            maze=self.sample_maze,
            start=self.sample_start,
            goal=self.sample_goal
        )

        # Mock user observation and assistant response
        messages = [
            {"role": "user", "content": "(1, 2): path, (1, 0): wall, (0, 1): wall, (2, 1): path"},
            {"role": "assistant", "content": "I'll move to (2, 1)"}
        ]

        with patch("verl.utils.reward_score.maze_mt.get_valid_moves", return_value=["(1, 2)", "(2, 1)"]):
            with patch("verl.utils.reward_score.maze_mt.extract_move", return_value=("(2, 1)", 2, 1)):
                with patch("verl.utils.reward_score.maze_mt.get_observation_chat", return_value=[{"role": "user", "content": "New observation"}]):
                    should_terminate, response, reward, metadata = await self.interaction.generate_response(instance_id, messages)

        assert should_terminate is False
        assert response == "New observation"
        assert reward == 0.0
        assert metadata == {}
        assert self.interaction._instance_dict[instance_id]["response"] == "I'll move to (2, 1)"

    @pytest.mark.asyncio
    async def test_generate_response_valid_move_reaches_goal(self):
        """Test generate_response with valid move that reaches goal."""
        instance_id = "test_instance"
        
        # Setup instance
        await self.interaction.start_interaction(
            instance_id=instance_id,
            maze=self.sample_maze,
            start=self.sample_start,
            goal=self.sample_goal
        )

        messages = [
            {"role": "user", "content": "(3, 2): path, (3, 4): wall, (2, 3): path, (4, 3): wall"},
            {"role": "assistant", "content": "I'll move to (3, 3)"}
        ]

        with patch("verl.utils.reward_score.maze_mt.get_valid_moves", return_value=["(3, 2)", "(2, 3)", "(3, 3)"]):
            with patch("verl.utils.reward_score.maze_mt.extract_move", return_value=("(3, 3)", 3, 3)):
                should_terminate, response, reward, metadata = await self.interaction.generate_response(instance_id, messages)

        assert should_terminate is True
        assert response == "Congratulations! You have reached the goal!"
        assert reward == 0.0
        assert metadata == {}

    @pytest.mark.asyncio
    async def test_generate_response_invalid_move(self):
        """Test generate_response with invalid move."""
        instance_id = "test_instance"
        
        # Setup instance
        await self.interaction.start_interaction(
            instance_id=instance_id,
            maze=self.sample_maze,
            start=self.sample_start,
            goal=self.sample_goal
        )

        messages = [
            {"role": "user", "content": "(1, 2): path, (1, 0): wall, (0, 1): wall, (2, 1): path"},
            {"role": "assistant", "content": "I'll move to (1, 0)"}
        ]

        with patch("verl.utils.reward_score.maze_mt.get_valid_moves", return_value=["(1, 2)", "(2, 1)"]):
            with patch("verl.utils.reward_score.maze_mt.extract_move", return_value=("(1, 0)", 1, 0)):
                should_terminate, response, reward, metadata = await self.interaction.generate_response(instance_id, messages)

        assert should_terminate is True
        assert response == "Your response is incorrect! Valid moves are: (1, 2), (2, 1). Please try again."
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_generate_response_multiple_messages(self):
        """Test generate_response with multiple messages (should use latest assistant and user messages)."""
        instance_id = "test_instance"
        
        # Setup instance
        await self.interaction.start_interaction(
            instance_id=instance_id,
            maze=self.sample_maze,
            start=self.sample_start,
            goal=self.sample_goal
        )

        messages = [
            {"role": "user", "content": "Old observation"},
            {"role": "assistant", "content": "Old response"},
            {"role": "user", "content": "(1, 2): path, (1, 0): wall, (0, 1): wall, (2, 1): path"},
            {"role": "assistant", "content": "I'll move to (2, 1)"}
        ]

        with patch("verl.utils.reward_score.maze_mt.get_valid_moves", return_value=["(1, 2)", "(2, 1)"]):
            with patch("verl.utils.reward_score.maze_mt.extract_move", return_value=("(2, 1)", 2, 1)):
                with patch("verl.utils.reward_score.maze_mt.get_observation_chat", return_value=[{"role": "user", "content": "New observation"}]):
                    should_terminate, response, reward, metadata = await self.interaction.generate_response(instance_id, messages)

        assert should_terminate is False
        assert self.interaction._instance_dict[instance_id]["response"] == "I'll move to (2, 1)"
        assert self.interaction._instance_dict[instance_id]["user_content"] == "(1, 2): path, (1, 0): wall, (0, 1): wall, (2, 1): path"

    @pytest.mark.asyncio
    async def test_generate_response_no_user_message(self):
        """Test generate_response with no user messages."""
        instance_id = "test_instance"
        
        # Setup instance
        await self.interaction.start_interaction(
            instance_id=instance_id,
            maze=self.sample_maze,
            start=self.sample_start,
            goal=self.sample_goal
        )

        messages = [{"role": "assistant", "content": "Hello!"}]

        with patch("verl.utils.reward_score.maze_mt.get_valid_moves", return_value=[]):
            with patch("verl.utils.reward_score.maze_mt.extract_move", return_value=(None, None, None)):
                should_terminate, response, reward, metadata = await self.interaction.generate_response(instance_id, messages)

        assert should_terminate is True
        assert reward == 0.0
        assert self.interaction._instance_dict[instance_id]["response"] == "Hello!"
        assert self.interaction._instance_dict[instance_id]["user_content"] == ""

    @pytest.mark.asyncio
    async def test_generate_response_observation_generation_fails(self):
        """Test generate_response when observation generation fails."""
        instance_id = "test_instance"
        
        # Setup instance
        await self.interaction.start_interaction(
            instance_id=instance_id,
            maze=self.sample_maze,
            start=self.sample_start,
            goal=self.sample_goal
        )

        messages = [
            {"role": "user", "content": "(1, 2): path, (2, 1): path"},
            {"role": "assistant", "content": "I'll move to (2, 1)"}
        ]

        with patch("verl.utils.reward_score.maze_mt.get_valid_moves", return_value=["(1, 2)", "(2, 1)"]):
            with patch("verl.utils.reward_score.maze_mt.extract_move", return_value=("(2, 1)", 2, 1)):
                with patch("verl.utils.reward_score.maze_mt.get_observation_chat", return_value=[]):
                    should_terminate, response, reward, metadata = await self.interaction.generate_response(instance_id, messages)

        assert should_terminate is False
        assert response == "Unable to generate observation."

    @pytest.mark.asyncio
    async def test_calculate_feedback_direct_call(self):
        """Test calculate_feedback method directly."""
        instance_id = "test_instance"
        
        # Setup instance
        await self.interaction.start_interaction(
            instance_id=instance_id,
            maze=self.sample_maze,
            start=self.sample_start,
            goal=self.sample_goal
        )

        # Set response and user content
        self.interaction._instance_dict[instance_id]["response"] = "I'll move to (2, 1)"
        self.interaction._instance_dict[instance_id]["user_content"] = "(1, 2): path, (2, 1): path"

        with patch("verl.utils.reward_score.maze_mt.get_valid_moves", return_value=["(1, 2)", "(2, 1)"]):
            with patch("verl.utils.reward_score.maze_mt.extract_move", return_value=("(2, 1)", 2, 1)):
                with patch("verl.utils.reward_score.maze_mt.get_observation_chat", return_value=[{"role": "user", "content": "New observation"}]):
                    should_terminate, response = await self.interaction.calculate_feedback(instance_id)

        assert should_terminate is False
        assert response == "New observation"

    @pytest.mark.asyncio
    async def test_finalize_interaction(self):
        """Test finalize_interaction method."""
        instance_id = "test_instance"
        
        # Setup instance
        await self.interaction.start_interaction(
            instance_id=instance_id,
            maze=self.sample_maze,
            start=self.sample_start,
            goal=self.sample_goal
        )

        assert instance_id in self.interaction._instance_dict

        await self.interaction.finalize_interaction(instance_id)

        assert instance_id not in self.interaction._instance_dict

    @pytest.mark.asyncio
    async def test_finalize_interaction_with_kwargs(self):
        """Test finalize_interaction method with additional kwargs."""
        instance_id = "test_instance"
        
        # Setup instance
        await self.interaction.start_interaction(
            instance_id=instance_id,
            maze=self.sample_maze,
            start=self.sample_start,
            goal=self.sample_goal
        )

        assert instance_id in self.interaction._instance_dict

        await self.interaction.finalize_interaction(instance_id, extra_param="test")

        assert instance_id not in self.interaction._instance_dict

    @pytest.mark.asyncio
    async def test_finalize_nonexistent_interaction(self):
        """Test finalize_interaction with non-existent instance_id."""
        instance_id = "nonexistent_instance"

        # This should raise KeyError
        with pytest.raises(KeyError):
            await self.interaction.finalize_interaction(instance_id)

    @pytest.mark.asyncio
    async def test_full_interaction_workflow_success(self):
        """Test complete interaction workflow with successful navigation."""
        # Start interaction
        instance_id = await self.interaction.start_interaction(
            maze=self.sample_maze,
            start=self.sample_start,
            goal=self.sample_goal
        )

        # First move (valid but not goal)
        messages = [
            {"role": "user", "content": "(1, 2): path, (2, 1): path"},
            {"role": "assistant", "content": "I'll move to (2, 1)"}
        ]

        with patch("verl.utils.reward_score.maze_mt.get_valid_moves", return_value=["(1, 2)", "(2, 1)"]):
            with patch("verl.utils.reward_score.maze_mt.extract_move", return_value=("(2, 1)", 2, 1)):
                with patch("verl.utils.reward_score.maze_mt.get_observation_chat", return_value=[{"role": "user", "content": "Next observation"}]):
                    should_terminate, response, reward, metadata = await self.interaction.generate_response(instance_id, messages)

        assert should_terminate is False
        assert reward == 0.0

        # Second move (reaches goal)
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": "Next observation"})
        messages.append({"role": "assistant", "content": "I'll move to (3, 3)"})

        with patch("verl.utils.reward_score.maze_mt.get_valid_moves", return_value=["(2, 2)", "(3, 3)"]):
            with patch("verl.utils.reward_score.maze_mt.extract_move", return_value=("(3, 3)", 3, 3)):
                should_terminate, response, reward, metadata = await self.interaction.generate_response(instance_id, messages)

        assert should_terminate is True
        assert response == "Congratulations! You have reached the goal!"

        # Finalize interaction
        await self.interaction.finalize_interaction(instance_id)
        assert instance_id not in self.interaction._instance_dict

    @pytest.mark.asyncio
    async def test_full_interaction_workflow_failure(self):
        """Test complete interaction workflow with invalid move."""
        # Start interaction
        instance_id = await self.interaction.start_interaction(
            maze=self.sample_maze,
            start=self.sample_start,
            goal=self.sample_goal
        )

        # Invalid move
        messages = [
            {"role": "user", "content": "(1, 2): path, (2, 1): path"},
            {"role": "assistant", "content": "I'll move to (1, 0)"}
        ]

        with patch("verl.utils.reward_score.maze_mt.get_valid_moves", return_value=["(1, 2)", "(2, 1)"]):
            with patch("verl.utils.reward_score.maze_mt.extract_move", return_value=("(1, 0)", 1, 0)):
                should_terminate, response, reward, metadata = await self.interaction.generate_response(instance_id, messages)

        assert should_terminate is True
        assert reward == 0.0
        assert "incorrect" in response.lower()

        # Finalize interaction
        await self.interaction.finalize_interaction(instance_id)
        assert instance_id not in self.interaction._instance_dict

    @pytest.mark.asyncio
    async def test_multiple_concurrent_interactions(self):
        """Test multiple concurrent interaction instances."""
        maze_1 = [[0, 0], [0, -1]]
        maze_2 = [[0, 1], [0, -1]]
        start_1 = (1, 1)
        start_2 = (1, 1)
        goal_1 = (2, 2)
        goal_2 = (2, 2)

        # Start multiple interactions
        instance_id_1 = await self.interaction.start_interaction(
            maze=maze_1, start=start_1, goal=goal_1
        )
        instance_id_2 = await self.interaction.start_interaction(
            maze=maze_2, start=start_2, goal=goal_2
        )

        assert len(self.interaction._instance_dict) == 2
        assert instance_id_1 in self.interaction._instance_dict
        assert instance_id_2 in self.interaction._instance_dict

        # Test responses for both instances
        messages_1 = [
            {"role": "user", "content": "(1, 2): path, (2, 1): path"},
            {"role": "assistant", "content": "I'll move to (2, 2)"}
        ]
        messages_2 = [
            {"role": "user", "content": "(1, 2): wall, (2, 1): path"},
            {"role": "assistant", "content": "I'll move to (2, 2)"}
        ]

        with patch("verl.utils.reward_score.maze_mt.get_valid_moves", side_effect=[["(1, 2)", "(2, 1)", "(2, 2)"], ["(2, 1)", "(2, 2)"]]):
            with patch("verl.utils.reward_score.maze_mt.extract_move", side_effect=[("(2, 2)", 2, 2), ("(2, 2)", 2, 2)]):
                should_terminate_1, _, reward_1, _ = await self.interaction.generate_response(instance_id_1, messages_1)
                should_terminate_2, _, reward_2, _ = await self.interaction.generate_response(instance_id_2, messages_2)

        assert should_terminate_1 is True
        assert should_terminate_2 is True
        assert reward_1 == 0.0
        assert reward_2 == 0.0

        # Finalize both interactions
        await self.interaction.finalize_interaction(instance_id_1)
        await self.interaction.finalize_interaction(instance_id_2)

        assert len(self.interaction._instance_dict) == 0

    @pytest.mark.asyncio
    async def test_edge_case_empty_messages(self):
        """Test edge case with empty messages list."""
        instance_id = "test_instance"
        
        # Setup instance
        await self.interaction.start_interaction(
            instance_id=instance_id,
            maze=self.sample_maze,
            start=self.sample_start,
            goal=self.sample_goal
        )

        messages = []

        with patch("verl.utils.reward_score.maze_mt.get_valid_moves", return_value=[]):
            with patch("verl.utils.reward_score.maze_mt.extract_move", return_value=(None, None, None)):
                should_terminate, response, reward, metadata = await self.interaction.generate_response(instance_id, messages)

        assert should_terminate is True
        assert reward == 0.0
        assert self.interaction._instance_dict[instance_id]["response"] == ""
        assert self.interaction._instance_dict[instance_id]["user_content"] == ""

    @pytest.mark.asyncio
    async def test_edge_case_message_without_content(self):
        """Test edge case with message without content field."""
        instance_id = "test_instance"
        
        # Setup instance
        await self.interaction.start_interaction(
            instance_id=instance_id,
            maze=self.sample_maze,
            start=self.sample_start,
            goal=self.sample_goal
        )

        messages = [
            {"role": "user"},  # Missing content field
            {"role": "assistant"}  # Missing content field
        ]

        with patch("verl.utils.reward_score.maze_mt.get_valid_moves", return_value=[]):
            with patch("verl.utils.reward_score.maze_mt.extract_move", return_value=(None, None, None)):
                should_terminate, response, reward, metadata = await self.interaction.generate_response(instance_id, messages)

        assert should_terminate is True
        assert reward == 0.0
        assert self.interaction._instance_dict[instance_id]["response"] is None
        assert self.interaction._instance_dict[instance_id]["user_content"] is None

    @pytest.mark.asyncio
    async def test_edge_case_goal_coordinate_conversion(self):
        """Test edge case with different goal coordinate systems."""
        instance_id = "test_instance"
        
        # Setup instance with 0-based goal coordinates
        await self.interaction.start_interaction(
            instance_id=instance_id,
            maze=self.sample_maze,
            start=(1, 1),
            goal=(2, 2)  # 0-based coordinate
        )

        messages = [
            {"role": "user", "content": "(3, 3): exit"},
            {"role": "assistant", "content": "I'll move to (3, 3)"}
        ]

        with patch("verl.utils.reward_score.maze_mt.get_valid_moves", return_value=["(3, 3)"]):
            with patch("verl.utils.reward_score.maze_mt.extract_move", return_value=("(3, 3)", 3, 3)):
                should_terminate, response, reward, metadata = await self.interaction.generate_response(instance_id, messages)

        # Should reach goal because 3 == 2 + 1 (converting 0-based to 1-based)
        assert should_terminate is True
        assert response == "Congratulations! You have reached the goal!"

    def test_inheritance_from_base_interaction(self):
        """Test that MazeInteraction properly inherits from BaseInteraction."""
        from verl.interactions.base import BaseInteraction

        assert isinstance(self.interaction, BaseInteraction)

        # Test that all required methods are implemented
        assert hasattr(self.interaction, "start_interaction")
        assert hasattr(self.interaction, "generate_response")
        assert hasattr(self.interaction, "calculate_score")
        assert hasattr(self.interaction, "finalize_interaction")

        # Test that methods are callable
        assert callable(self.interaction.start_interaction)
        assert callable(self.interaction.generate_response)
        assert callable(self.interaction.calculate_score)
        assert callable(self.interaction.finalize_interaction)

    def test_maze_specific_methods(self):
        """Test maze-specific helper methods."""
        # Test that maze-specific methods are present
        assert hasattr(self.interaction, "_parse_maze_string")
        assert hasattr(self.interaction, "_parse_coordinate_string")
        assert hasattr(self.interaction, "calculate_feedback")

        # Test that methods are callable
        assert callable(self.interaction._parse_maze_string)
        assert callable(self.interaction._parse_coordinate_string)
        assert callable(self.interaction.calculate_feedback)

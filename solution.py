import gymnasium as gym
import numpy as np
import logging
import sys
import tty
import termios


from arenax_sai import SAIClient
from stable_baselines3 import PPO
from collections import deque
from typing import List, Tuple
from gymnasium import Wrapper
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, SupportsFloat, TypeVar

from copy import deepcopy

from gymnasium import logger, spaces
from gymnasium.utils import RecordConstructorArgs, seeding

from gymnasium.envs.registration import EnvSpec, WrapperSpec

from gymnasium.core import (
    Env,
    WrapperObsType, 
    WrapperActType,
    Generic, 
    ObsType, 
    ActType,
    RenderFrame
)


class State(Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    INVALID = -1


class Mouse(Enum):
    BOTTOM = 0
    TOP = 1


class Section(Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    INVALID = -1


class Door(Enum):
    ONE = 1
    TWO = 2
    THREE = 3


def get_key():
    """
    Get a single keypress from the user without requiring Enter.
    Specifically for macOS.
    Returns the character pressed as a string.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def find_shortest_distance(maze, start: Tuple[int, int], end: Tuple[int, int]) -> int:
    """
    Find the shortest distance from start to end in a binary maze using BFS.
    Includes diagonal movement.
    
    Args:
        maze: 2D list where 0 represents path and 1 represents wall
        start: Starting coordinates (row, col)
        end: Target coordinates (row, col)
    
    Returns:
        Integer representing shortest distance (-1 if no path exists)
    """
    rows, cols = len(maze[:, 0]), len(maze[0, :])
    
    # Check if start or end positions are valid
    if (not (0 <= start[0] < rows and 0 <= start[1] < cols) or
        not (0 <= end[0] < rows and 0 <= end[1] < cols) or
        maze[start[0], start[1]] == 1 or maze[end[0], end[1]] == 1):
        return -1
    
    # Possible movements: up, right, down, left + diagonals
    directions = [
        (-1, 0),  # up
        (-1, 1),  # up-right
        (0, 1),   # right
        (1, 1),   # down-right
        (1, 0),   # down
        (1, -1),  # down-left
        (0, -1),  # left
        (-1, -1)  # up-left
    ]
    
    # Queue for BFS: (row, col, distance)
    queue = deque([(start[0], start[1], 0)])
    visited = {start}
    
    while queue:
        row, col, dist = queue.popleft()
        
        if (row, col) == end:
            return dist
        
        # Explore all possible directions including diagonals
        for dx, dy in directions:
            new_row, new_col = row + dx, col + dy
            new_pos = (new_row, new_col)
            
            if (0 <= new_row < rows and 
                0 <= new_col < cols and 
                maze[new_row, new_col] == 0 and 
                new_pos not in visited):
                queue.append((new_row, new_col, dist + 1))
                visited.add(new_pos)
    
    return -1  # No path found




class RewardShapingWrapper(Wrapper):

    def __init__(self, env, movement_reward=1, incorrect_mouse_reward=-0.5):
        super().__init__(env)
        self.door_channel_map = {
            Door.ONE: 7,
            Door.TWO: 8,
            Door.THREE: 9
        }
        self.current_mouse = Mouse.TOP
        self.previous_distance_to_target = None
        self.cumulative_reward = 0
        self.previous_observation = None
        self.previous_action = None
        self.previous_state = None
        self.movement_reward = movement_reward
        self.incorrect_mouse_reward = incorrect_mouse_reward

        self.state_conditions = {
            State.ONE: [
                lambda observation: (self.current_section(observation, Mouse.TOP) == Section.ONE),
                lambda observation: (self.current_section(observation, Mouse.BOTTOM) == Section.FIVE),
                lambda observation: (not self.door_is_open(observation, Door.THREE)),
            ],
            State.TWO: [
                lambda observation: self.door_is_open(observation, Door.THREE),
                lambda observation: (not self.door_is_open(observation, Door.TWO)),
                lambda observation: (self.current_section(observation, Mouse.TOP) == Section.ONE),
                lambda observation: (self.current_section(observation, Mouse.BOTTOM) in [Section.FOUR, Section.FIVE])
            ],
            State.THREE: [
                lambda observation: (not self.door_is_open(observation, Door.THREE)),
                lambda observation: (self.door_is_open(observation, Door.TWO)),
                lambda observation: (self.current_section(observation, Mouse.TOP) == Section.ONE),
                lambda observation: (self.current_section(observation, Mouse.BOTTOM) in [Section.THREE, Section.FOUR])
            ],
            State.FOUR: [
                lambda observation: (self.current_section(observation, Mouse.BOTTOM) == Section.THREE),
                lambda observation: (self.current_section(observation, Mouse.TOP) in [Section.ONE, Section.TWO]),
                lambda observation: (self.door_is_open(observation, Door.ONE)),
                lambda observation: (not self.door_is_open(observation, Door.TWO))
            ],
            State.FIVE: [
                lambda observation: (self.door_is_open(observation, Door.TWO)),
                lambda observation: (self.current_section(observation, Mouse.BOTTOM) == Section.THREE),
                lambda observation: (self.current_section(observation, Mouse.TOP) in [Section.THREE, Section.TWO]),
                lambda observation: (self.mouse_current_position(observation, Mouse.TOP) != (9, 6)),
            ],
            State.SIX: [
                lambda observation: (self.door_is_open(observation, Door.TWO)),
                lambda observation: (self.current_section(observation, Mouse.BOTTOM) == Section.THREE),
                lambda observation: (self.current_section(observation, Mouse.TOP) in [Section.THREE, Section.TWO]),
                lambda observation: (self.mouse_current_position(observation, Mouse.TOP) == (9, 6)),
                lambda observation: (self.mouse_current_position(observation, Mouse.BOTTOM) != (9, 6)),
            ],
            State.SEVEN: [
                lambda observation: self.door_is_open(observation, Door.THREE),
                lambda observation: (self.current_section(observation, Mouse.BOTTOM) == Section.THREE),
                lambda observation: (self.current_section(observation, Mouse.TOP) in [Section.ONE, Section.TWO]),
            ]
        }
        self.reward_shaping_functions = {
            State.ONE: [
                lambda observation: self.correct_mouse_reward(Mouse.TOP),
                lambda observation: self.distance_to_target_delta_reward(observation, Mouse.TOP, (1,1))
            ],
            State.TWO: [
                lambda observation: self.correct_mouse_reward(Mouse.BOTTOM),
                lambda observation: self.distance_to_target_delta_reward(observation, Mouse.BOTTOM, (13, 1)) 
            ],
            State.THREE: [
                lambda observation: self.correct_mouse_reward(Mouse.BOTTOM),
                lambda observation: self.distance_to_target_delta_reward(observation, Mouse.BOTTOM, (9, 11)) 
            ],
            State.FOUR: [
                lambda observation: self.correct_mouse_reward(Mouse.TOP),
                lambda observation: self.distance_to_target_delta_reward(observation, Mouse.TOP, (5, 11)),
                lambda observation: -self.distance_to_target_delta_reward(observation, Mouse.TOP, (1, 1))/2
            ],
            State.FIVE: [
                lambda observation: self.correct_mouse_reward(Mouse.TOP),
                lambda observation: self.distance_to_target_delta_reward(observation, Mouse.TOP, (9, 6))
            ],
            State.SIX: [
                lambda observation: self.correct_mouse_reward(Mouse.BOTTOM),
                lambda observation: self.distance_to_target_delta_reward(observation, Mouse.BOTTOM, (9, 6))
            ],
            State.SEVEN: [
                lambda observation: self.correct_mouse_reward(Mouse.BOTTOM),
                lambda observation: self.distance_to_target_delta_reward(observation, Mouse.BOTTOM, (9, 11)) 
            ]
        }


    def construct_maze(self, observation):
        maze = observation[:, :, 1]
        for door_channel_index in range(7, 12):
            door_channel = observation[:, :, door_channel_index]
            maze -= (door_channel == -1).astype(int)
        maze *= -1
        return maze


    def get_current_position(self, maze):
        num_rows = len(maze)
        num_columns = len(maze[0])
        for i in range(num_rows):
            for j in range(num_columns):
                if maze[i, j] == 1:
                    return (i,j)

    
    def distance_to_target_delta_reward(self, observation, mouse, target):
        maze = self.construct_maze(observation)
        source = self.mouse_current_position(observation, mouse)
        if self.previous_distance_to_target is None or self.previous_distance_to_target == 0:
            self.previous_distance_to_target = find_shortest_distance(maze, source, target)
            return 0
        else:
            current_distance_to_target = find_shortest_distance(maze, source, target)
            logging.info(f"Distance of source {source} to target {target}: {current_distance_to_target}")
            delta = self.previous_distance_to_target - current_distance_to_target
            self.previous_distance_to_target = current_distance_to_target
            return delta


    def correct_mouse_reward(self, desired_mouse):
        if self.current_mouse == desired_mouse:
            return 0
        else:
            return self.incorrect_mouse_reward


    def shape_reward(self, observation, reward, action, current_state):
        shaped_reward = reward
        for shaping_function in self.reward_shaping_functions[current_state]:
            shaped_reward += shaping_function(observation)
        return shaped_reward


    def door_is_open(self, observation, door):
        channel_index = self.door_channel_map[door]
        door_channel = observation[:, :, channel_index]
        return np.max(door_channel) == 1


    def current_section(self, observation, mouse):
        position = self.mouse_current_position(observation, mouse)
        if position is None:
            return Section.INVALID
        row, _ = position
        if row < 3:
            return Section.ONE
        elif row < 7:
            return Section.TWO
        elif row < 11:
            return Section.THREE
        elif row < 15:
            return Section.FOUR
        else:
            return Section.FIVE


    def determine_current_mouse(self, action):
        if action != 9 or self.previous_action == 9:
            current_mouse = self.current_mouse
        else:
            if self.current_mouse == Mouse.TOP:
                current_mouse = Mouse.BOTTOM
            else:
                current_mouse = Mouse.TOP
        logging.info(f"Current mouse: {current_mouse}")
        return current_mouse


    def mouse_current_position(self, observation, mouse):
        if mouse == self.current_mouse:
            mouse_factor = 1
        else:
            mouse_factor = -1
        current_map = mouse_factor*observation[:, :, 0]
        position = self.get_current_position(current_map)
        return position 


    def determine_current_state(self, observation, action):
        self.current_mouse = self.determine_current_mouse(action)
        current_states = [
            state
            for (state, conditions) in self.state_conditions.items()
            if all(condition(observation) for condition in conditions)
        ]
        logging.debug(f"Current states: {current_states}")
        self.previous_action = action
        self.previous_observation = observation
        if len(current_states) != 1:
            return State.INVALID
        else:
            return current_states[0]


    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        current_state = self.determine_current_state(observation, action)
        logging.info(f"State: {current_state}")
        if current_state == State.INVALID:
            shaped_reward = 0
            self.cumulative_reward = 0
            terminated = True
            for i, condition in enumerate(self.state_conditions[self.previous_state]):
                if not condition(observation):
                    logging.error(f"Invalid condition for state {self.previous_state}: {i}")
            raise Exception("Entered invalid state")
        else:
            shaped_reward = self.shape_reward(observation, reward, action, current_state)
            self.cumulative_reward += shaped_reward
            terminated = False
            truncated = False
        logging.info(f"Current reward: {shaped_reward}")
        logging.info(f"Cumulative reward: {self.cumulative_reward}")
        logging.debug(f"Door channels:\n")
        for door, channel_index in self.door_channel_map.items():
            channel = observation[:, :, channel_index]
            logging.debug(f"Door {door}:\n{channel}")
        for i in range(3):
            channel_index = 2 + i
            channel = observation[:, :, channel_index]
            logging.debug(f"Switch {i+1}:\n{channel}")
        self.previous_state = current_state
        return observation, shaped_reward, terminated, truncated, info


    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        self.previous_distance_to_target = None
        self.cumulative_reward = 0
        self.previous_state = State.ONE
        self.env.reset()
        observation, _, _, _, info = self.env.step(3)
        return observation, info

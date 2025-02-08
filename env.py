import logging
import sys
import tty
import termios
import pickle
import wandb
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn

from copy import deepcopy
from arenax_sai import SAIClient
from collections import deque
from typing import List, Tuple, Optional
from gymnasium import Wrapper
from enum import Enum
from functools import total_ordering
from typing import (
    TYPE_CHECKING, 
    Any, 
    Generic, 
    SupportsFloat, 
    TypeVar,
    Any, 
    Dict, 
    List, 
    Optional, 
    Tuple, 
    Type, 
    Union
)
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


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


@total_ordering
class State(Enum):
    INVALID = -2
    RESET = -1
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    WIN = 7
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


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


class PositionCounter:
    def __init__(self, position, count=1):
        self.position: (int, int) = position
        self.count: int = count


class ColumnCounter:
    def __init__(self, column, count=1):
        self.column: int = column
        self.count: int = count


ACTION_DIRECTION_MAP = {
    "N": 3,
    "NE": 2,
    "E": 8,
    "SE": 5,
    "S": 6,
    "SW": 4,
    "W": 7,
    "NW": 1
}

DOOR_CHANNEL_MAP = {
    Door.ONE: 7,
    Door.TWO: 8,
    Door.THREE: 9
}


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


def find_shortest_path(maze, start: Tuple[int, int], target: Tuple[int, int]):
    """
    Find the first step direction and shortest distance to target from start position using BFS.
    All moves (including diagonal) count as distance 1.
    
    Args:
        maze: Binary matrix where 0 represents path and 1 represents wall
        start: Starting coordinates (row, col)
        target: Target coordinates (row, col)
    
    Returns:
        Tuple of (direction, distance) where:
        - direction is one of 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW' or None if no path exists
        - distance is the number of steps in the shortest path or None if no path exists
    """
    rows, cols = len(maze[:, 0]), len(maze[0, :])
    
    # Validate input coordinates
    if (not (0 <= start[0] < rows and 0 <= start[1] < cols) or
        not (0 <= target[0] < rows and 0 <= target[1] < cols) or
        maze[start[0], start[1]] == 1 or maze[target[0], target[1]] == 1):
        return None, None
    
    # Direction vectors for N, NE, E, SE, S, SW, W, NW
    directions = [
        (-1, 0),  # N
        (-1, 1),  # NE
        (0, 1),   # E
        (1, 1),   # SE
        (1, 0),   # S
        (1, -1),  # SW
        (0, -1),  # W
        (-1, -1)  # NW
    ]
    direction_names = [
        'N', 
        'NE', 
        'E', 
        'SE', 
        'S', 
        'SW', 
        'W', 
        'NW'
    ]
    
    # Queue for BFS: (row, col, path, distance)
    queue = deque([(start[0], start[1], [], 0)])
    visited = {(start[0], start[1])}
    
    while queue:
        row, col, path, dist = queue.popleft()
        
        # If we reached the target, return the first step we took and the distance
        if (row, col) == target:
            return path[0] if path else None, dist
        
        # Try all eight directions
        for i, (dr, dc) in enumerate(directions):
            new_row, new_col = row + dr, col + dc
            
            # Check if the new position is valid
            if (0 <= new_row < rows and 
                0 <= new_col < cols and 
                maze[new_row, new_col] == 0 and 
                (new_row, new_col) not in visited):
                
                # For diagonal moves, check if both adjacent cells are clear
                is_diagonal = dr != 0 and dc != 0
                if is_diagonal:
                    if maze[row, new_col] == 1 or maze[new_row, col] == 1:
                        continue
                
                # Add direction to path if this is the first step
                new_path = path if path else [direction_names[i]]
                
                # All moves count as distance 1
                new_dist = dist + 1
                
                visited.add((new_row, new_col))
                queue.append((new_row, new_col, new_path, new_dist))
    
    return None, None


class RewardShapingWrapper(Wrapper):

    def __init__(self, env, movement_reward=1, incorrect_mouse_penalty=0.01, 
                 correct_direction_reward=0.25, state_transition_reward=1, 
                 win_reward_factor=10, position_penalty=0.01, 
                 save_dir='/var/tmp', passthrough=False, use_wandb=False, 
                 wandb_project="Co-op Puzzle", maze_path=None):

        super().__init__(env)
        self.current_mouse = None
        self.previous_distance_to_target = {
            Mouse.TOP: None,
            Mouse.BOTTOM: None
        }
        self.cumulative_reward = 0
        self.cumulative_state_reward = 0
        self.cumulative_directional_award = {
            Mouse.TOP: 0,
            Mouse.BOTTOM: 0
        }
        self.previous_observation = None
        self.previous_action = None
        self.previous_state = None
        self.previous_info = None
        self.save_dir = save_dir
        self.passthrough = passthrough
        self.previous_states = set()
        self.mouse_current_positions = {
            Mouse.TOP: None,
            Mouse.BOTTOM: None
        }
        self.previous_mouse = None

        self.mouse_current_columns = {
            mouse: None
            for mouse in [Mouse.TOP, Mouse.BOTTOM]
        }
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(
                project=wandb_project
            )

        if maze_path is not None:
            self.maze = np.loadtxt(maze_path, delimiter=',')
        else:
            self.maze = None

        self.movement_reward = movement_reward
        self.incorrect_mouse_penalty = incorrect_mouse_penalty
        self.correct_direction_reward = correct_direction_reward
        self.state_transition_reward = state_transition_reward
        self.win_reward_factor = win_reward_factor
        self.position_penalty = position_penalty

        self.state_conditions = {
            State.ZERO: [
                lambda observation: (self.current_section(observation, Mouse.TOP) in [Section.ONE]),
                lambda observation: (self.current_section(observation, Mouse.BOTTOM) in [Section.FIVE]),
                lambda observation: (
                    self.door_is_open(observation, Door.THREE) and self.previous_states == {State.RESET}
                    and self.previous_state != State.ONE
                )
            ],
            State.ONE: [
                lambda observation: (self.current_section(observation, Mouse.TOP) in [Section.ONE]),
                lambda observation: (self.current_section(observation, Mouse.BOTTOM) in [Section.FIVE]),
                lambda observation: (not self.door_is_open(observation, Door.THREE))
            ],
            State.TWO: [
                lambda observation: (
                    self.door_is_open(observation, Door.THREE) 
                    and (
                        self.previous_state in {State.ONE, State.TWO, State.THREE}
                        or self.current_section(observation, Mouse.BOTTOM) == Section.FOUR
                    )
                ),
                lambda observation: (not self.door_is_open(observation, Door.TWO)),
                lambda observation: (self.current_section(observation, Mouse.TOP) == Section.ONE),
                lambda observation: (self.current_section(observation, Mouse.BOTTOM) in [Section.FOUR, Section.FIVE])
            ],
            State.THREE: [
                lambda observation: (self.current_section(observation, Mouse.TOP) == Section.ONE),
                lambda observation: (self.current_section(observation, Mouse.BOTTOM) in [Section.THREE, Section.FOUR]),
                lambda observation: not self.door_is_open(observation, Door.ONE),
                lambda observation: (
                    self.door_is_open(observation, Door.TWO)
                    or self.current_section(observation, Mouse.BOTTOM) == Section.THREE
                ),
                lambda observation: (
                    not self.door_is_open(observation, Door.THREE) 
                    or self.current_section(observation, Mouse.BOTTOM) == Section.THREE
                ),
            ],
            State.FOUR: [
                lambda observation: (self.current_section(observation, Mouse.TOP) in [Section.ONE, Section.TWO]),
                lambda observation: (self.current_section(observation, Mouse.BOTTOM) == Section.THREE),
                lambda observation: (self.door_is_open(observation, Door.ONE)),
                lambda observation: (not self.door_is_open(observation, Door.TWO))
            ],
            State.FIVE: [
                lambda observation: (self.door_is_open(observation, Door.TWO)),
                lambda observation: (self.current_section(observation, Mouse.BOTTOM) == Section.THREE),
                lambda observation: (self.current_section(observation, Mouse.TOP) in [Section.THREE, Section.TWO]),
                lambda observation: (self.get_mouse_current_position(observation, Mouse.TOP) != (9, 6)),
            ],
            State.SIX: [
                lambda observation: (self.door_is_open(observation, Door.TWO)),
                lambda observation: (self.current_section(observation, Mouse.BOTTOM) == Section.THREE),
                lambda observation: (self.current_section(observation, Mouse.TOP) in [Section.THREE, Section.TWO]),
                lambda observation: (self.get_mouse_current_position(observation, Mouse.TOP) == (9, 6)),
                lambda observation: (self.get_mouse_current_position(observation, Mouse.BOTTOM) != (9, 6)),
            ],
            State.WIN: [
                lambda observation: (self.get_mouse_current_position(observation, Mouse.TOP) == (9, 6)),
                lambda observation: (self.get_mouse_current_position(observation, Mouse.BOTTOM) == (9, 6)),
            ]
        }

        self.oracle_policy = {
            State.INVALID: {
                "mouse": Mouse.TOP,
                "target": (1, 1)
            },
            State.ZERO: {
                "mouse": Mouse.BOTTOM,
                "target": (17, 11)
            },
            State.ONE: {
                "mouse": Mouse.TOP,
                "target": (1, 1)
            },
            State.TWO: {
                "mouse": Mouse.BOTTOM,
                "target": (13, 1)
            },
            State.THREE: {
                "mouse": Mouse.BOTTOM,
                "target": (9, 11)
            },
            State.FOUR: {
                "mouse": Mouse.TOP,
                "target": (5, 11)
            },
            State.FIVE: {
                "mouse": Mouse.TOP,
                "target": (9, 6)
            },
            State.SIX: {
                "mouse": Mouse.BOTTOM,
                "target": (9, 6)
            }
        }

        self.reward_functions = [
            #lambda observation: self.stagnation_penalty(Mouse.TOP, threshold=20),
            #lambda observation: self.stagnation_penalty(Mouse.BOTTOM, threshold=20),
            #lambda observation: self.column_stagnation_penalty(Mouse.TOP, column=0, threshold=20),
            #lambda observation: self.column_stagnation_penalty(Mouse.BOTTOM, column=0, threshold=20),
            #lambda observation: self.column_stagnation_penalty(Mouse.TOP, column=12, threshold=20),
            #lambda observation: self.column_stagnation_penalty(Mouse.BOTTOM, column=12, threshold=20),
        ]

        self.state_reward_functions = {
            State.ZERO: [
                #lambda observation: self.incorrect_mouse_penalty_reward(Mouse.BOTTOM),
                #lambda observation: self.distance_to_target_delta_reward(observation, Mouse.TOP, (1,6)),
                #lambda observation: self.distance_to_target_delta_reward(observation, Mouse.TOP, (1,1)),
                lambda observation: self.distance_to_target_delta_reward(observation, Mouse.BOTTOM, (17,11)),
                #lambda observation: self.distance_to_target_delta_reward(observation, Mouse.BOTTOM, (13, 1)), 
            ],
            State.ONE: [
                #lambda observation: self.incorrect_mouse_penalty_reward(Mouse.TOP),
                lambda observation: self.distance_to_target_delta_reward(observation, Mouse.TOP, (1,1)),
                #lambda observation: self.distance_to_target_delta_reward(observation, Mouse.BOTTOM, (17,3)),
            ],
            State.TWO: [
                #lambda observation: self.incorrect_mouse_penalty_reward(Mouse.BOTTOM),
                lambda observation: self.distance_to_target_delta_reward(observation, Mouse.BOTTOM, (13, 1)), 
                #lambda observation: self.distance_to_target_delta_reward(observation, Mouse.TOP, (1,6)),
            ],
            State.THREE: [
                #lambda observation: self.incorrect_mouse_penalty_reward(Mouse.BOTTOM),
                lambda observation: self.distance_to_target_delta_reward(observation, Mouse.BOTTOM, (9, 11)), 
                #lambda observation: self.distance_to_target_delta_reward(observation, Mouse.TOP, (1,11)),
            ],
            State.FOUR: [
                #lambda observation: self.incorrect_mouse_penalty_reward(desired_mouse=Mouse.TOP),
                #lambda observation: self.correct_mouse_reward(desired_mouse=Mouse.TOP),
                lambda observation: self.distance_to_target_delta_reward(observation, Mouse.TOP, (5, 11)),
                #lambda observation: self.distance_to_target_delta_reward(observation, Mouse.BOTTOM, (9, 6)),
            ],
            State.FIVE: [
                #lambda observation: self.incorrect_mouse_penalty_reward(Mouse.TOP),
                lambda observation: self.distance_to_target_delta_reward(observation, Mouse.TOP, (9, 6)),
                lambda observation: self.distance_to_target_delta_reward(observation, Mouse.BOTTOM, (9, 6))
            ],
            State.SIX: [
                #lambda observation: self.incorrect_mouse_penalty_reward(Mouse.BOTTOM),
                lambda observation: self.distance_to_target_delta_reward(observation, Mouse.TOP, (9, 6)),
                lambda observation: self.distance_to_target_delta_reward(observation, Mouse.BOTTOM, (9, 6))
            ],
            State.WIN: [

            ]
        }

        self.action_reward_functions = {
            State.ZERO: [
                #lambda action, observation: self.direction_to_target_reward(action, observation, Mouse.BOTTOM, (17,11)),
            ],
            State.ONE: [
                #lambda action, observation: self.direction_to_target_reward(action, observation, Mouse.TOP, (1,1)),
            ],
            State.TWO: [
                #lambda action, observation: self.direction_to_target_reward(action, observation, Mouse.BOTTOM, (13, 1)) 
            ],
            State.THREE: [
                #lambda action, observation: self.direction_to_target_reward(action, observation, Mouse.BOTTOM, (9, 11)), 
            ],
            State.FOUR: [
                #lambda action, observation: self.direction_to_target_reward(action, observation, Mouse.TOP, (5, 11)),
            ],
            State.FIVE: [
                #lambda action, observation: self.direction_to_target_reward(action, observation, Mouse.TOP, (9, 6)),
            ],
            State.SIX: [
                #lambda action, observation: self.direction_to_target_reward(action, observation, Mouse.BOTTOM, (9, 6))
            ],
            State.WIN: [

            ]
        }

        self.state_transition_reward_functions = {
            State.ZERO: [
                lambda state, previous_state: 10*self.compute_state_transition_reward(state, previous_state)
            ],
            State.ONE: [
                lambda state, previous_state: 10*self.compute_state_transition_reward(state, previous_state)
            ],
            State.TWO: [
                lambda state, previous_state: 10*self.compute_state_transition_reward(state, previous_state)
            ],
            State.THREE: [
                lambda state, previous_state: 10*self.compute_state_transition_reward(state, previous_state)
            ],
            State.FOUR: [
                lambda state, previous_state: 10*self.compute_state_transition_reward(state, previous_state)
            ],
            State.FIVE: [
                lambda state, previous_state: 10*self.compute_state_transition_reward(state, previous_state)
            ],
            State.SIX: [
                lambda state, previous_state: 10*self.compute_state_transition_reward(state, previous_state)
            ],
            State.WIN: [
                lambda state, previous_state: 10*self.win_reward_factor*self.compute_state_transition_reward(state, previous_state)
            ]
        }


    def wandb_log(self, metrics: dict):
        if self.use_wandb:
            wandb.log(metrics)


    def construct_maze(self, observation, maze=None):
        if maze is None:
            maze = deepcopy(observation[:, :, 1])
        else:
            maze = deepcopy(maze)
        for door_channel_index in range(7, 12):
            door_channel = deepcopy(observation[:, :, door_channel_index])
            maze = np.minimum(maze, -(door_channel == -1).astype(int))
        maze *= -1
        return maze


    def get_current_position(self, maze):
        num_rows = len(maze)
        num_columns = len(maze[0])
        current_position = None
        for i in range(num_rows):
            for j in range(num_columns):
                if maze[i, j] == 1:
                    current_position = (i,j)
        if current_position is None:
            for i in range(num_rows):
                for j in range(num_columns):
                    if maze[i, j] == -1:
                        current_position = (i,j)
        return current_position

        
    def distance_to_target_delta_reward(self, observation, mouse, target):
        maze = self.construct_maze(observation)
        source = self.get_mouse_current_position(observation, mouse)
        reward = 0

        if self.previous_distance_to_target[mouse] is None:
            _, distance = find_shortest_path(maze, source, target)
            self.previous_distance_to_target[mouse] = distance
            reward = 0

        elif source is not None and target is not None and source == target:
            self.previous_distance_to_target[mouse] = 0
            reward = 0

        else:
            _, current_distance_to_target = find_shortest_path(maze, source, target)
            logging.debug(f"Distance of source {source} to target {target}: {current_distance_to_target}")

            try:
                delta = self.previous_distance_to_target[mouse] - current_distance_to_target
                self.previous_distance_to_target[mouse] = current_distance_to_target
                sign = 1 if delta >= 0 else -1
                abs_val = np.abs(delta)
                reward = sign*min(abs_val, 1)
            except:
                reward = 0

        reward_label = f"Reward - distance to target, {mouse}, target {target}"
        self.wandb_log({
            reward_label: reward
        })
        logging.debug(f"{reward_label}: {reward}")
        return reward


    def direction_to_target_reward(self, action, observation, mouse, target):
        if observation is not None and action in ACTION_DIRECTION_MAP.values() and mouse == self.current_mouse:
            maze = self.construct_maze(observation)
            source = self.get_mouse_current_position(observation, mouse)
            direction_to_target, _ = find_shortest_path(maze, source, target)
            if direction_to_target is not None and action == ACTION_DIRECTION_MAP[direction_to_target]:
                self.cumulative_directional_award[mouse] += self.correct_direction_reward
                reward = self.correct_direction_reward
            else:
                reward = -self.cumulative_directional_award[mouse]
                self.cumulative_directional_award[mouse] = 0
                reward = reward
        else:
            reward = 0
        reward_label = f"Reward - Direction to Target, {mouse}, target {target}"
        self.wandb_log({
            reward_label: reward
        })
        logging.debug(f"{reward_label}: {reward}")
        return reward


    def stagnation_penalty(self, mouse, threshold=20):
        if self.current_mouse == mouse and self.mouse_current_positions[mouse].count > threshold:
            reward = -(1+self.position_penalty)**(self.mouse_current_positions[mouse].count - threshold) + 1
        else:
            reward = 0
        reward_label = f"Reward - Stagnation Penalty, {mouse}, threshold {threshold}"
        self.wandb_log({
            reward_label: reward
        })
        logging.debug(f"{reward_label}: {reward}")
        return reward


    def position_stagnation_penalty(self, mouse, position, threshold=20):
        if self.current_mouse == mouse \
            and self.mouse_current_positions[mouse].position == position \
            and self.mouse_current_positions[mouse].count > threshold:

            reward = -(1+self.position_penalty)**(self.mouse_current_positions[mouse].count - threshold) + 1
        else:
            reward = 0
        reward_label = "Reward - Position Stagnation Penalty, {mouse}, position {position}"
        self.wandb_log({
            f"{reward_label}": reward
        })
        logging.debug(f"{reward_label}: {reward}")
        return reward


    def column_stagnation_penalty(self, mouse, column, threshold=20):
        if self.current_mouse == mouse \
            and self.mouse_current_columns[mouse].column == column \
            and self.mouse_current_columns[mouse].count > threshold:

            reward = -(1+self.position_penalty)**(self.mouse_current_columns[mouse].count - threshold) + 1
        else:
            reward = 0
        reward_label = f"Reward - Column Stagnation Penalty, {mouse}, column {column}"
        self.wandb_log({
            reward_label: reward
        })
        logging.debug(f"{reward_label}: {reward}")
        return reward


    def mouse_movement_delta(self, observation, previous_observation, mouse):
        current_location = self.get_mouse_current_position(observation, mouse)
        previous_location = self.get_mouse_current_position(previous_observation, mouse)
        distance = np.abs(current_location[0] - previous_location[0]) + np.abs(current_location[1] - previous_location[1])
        return distance


    def incorrect_mouse_penalty_reward(self, desired_mouse):
        if self.current_mouse == desired_mouse:
            reward = 0
        else:
            reward = -self.incorrect_mouse_penalty
        reward_label = f"Reward - Incorrect Mouse Penalty, {desired_mouse}"
        self.wandb_log({
            reward_label: reward
        })
        logging.debug(f"{reward_label}: {reward}")
        return reward


    def correct_mouse_reward(self, desired_mouse):
        if self.current_mouse == desired_mouse:
            reward = self.incorrect_mouse_penalty
        else:
            reward = 0
        reward_label = f"Reward - Correct Mouse Reward, {desired_mouse}"
        self.wandb_log({
            reward_label: reward
        })
        logging.debug(f"{reward_label}: {reward}")
        return reward


    def compute_state_transition_reward(self, state, previous_state):
        if state > previous_state:
            reward = self.state_transition_reward
        elif state < previous_state:
            reward = -self.state_transition_reward
        else:
            reward = 0
        reward_label = f"Reward - State Transition, {previous_state} -> {state}"
        self.wandb_log({
            reward_label: reward
        })
        logging.debug(f"{reward_label}: {reward}")
        return reward


    def shape_reward(self, observation, reward, action, current_state, previous_state, terminated):
        shaped_reward = reward

        if terminated and not current_state == State.WIN:
            shaped_reward -= self.cumulative_reward if self.cumulative_reward > 0 else 0

        else:
            for shaping_function in self.reward_functions:
                shaped_reward += shaping_function(observation)

            try:
                for shaping_function in self.state_reward_functions[current_state]:
                    shaped_reward += shaping_function(observation)
            except KeyError:
                pass
            try:
                for shaping_function in self.action_reward_functions[current_state]:
                    shaped_reward += shaping_function(action, self.previous_observation)
            except KeyError:
                pass
            try:
                for shaping_function in self.state_transition_reward_functions[current_state]:
                    shaped_reward += shaping_function(current_state, previous_state)
            except KeyError:
                pass

        return shaped_reward


    def door_is_open(self, observation, door):
        channel_index = DOOR_CHANNEL_MAP[door]
        door_channel = observation[:, :, channel_index]
        return np.max(door_channel) == 1


    def current_section(self, observation, mouse):
        position = self.get_mouse_current_position(observation, mouse)
        if position is None:
            section =  Section.INVALID
        else:
            row, _ = position
            if row <= 3:
                section = Section.ONE
            elif row <= 7:
                section = Section.TWO
            elif row <= 11:
                section = Section.THREE
            elif row <= 15:
                section = Section.FOUR
            else:
                section = Section.FIVE
        logging.debug(f"Mouse {mouse} in section {section}")
        return section


    def determine_current_mouse(self, observation, action, info):
        if info["timestep"] == 1:
            if observation[1, 6, 0] == 1:
                current_mouse = Mouse.TOP
            else:
                current_mouse = Mouse.BOTTOM
        elif info["timestep"] == 2 and action == 9:
            current_mouse = Mouse.BOTTOM
        elif info["timestep"] == 2 and action != 9:
            current_mouse = Mouse.TOP
        elif action != 9 or self.previous_action == 9:
            current_mouse = self.current_mouse
        else:
            if self.current_mouse == Mouse.TOP:
                current_mouse = Mouse.BOTTOM
            else:
                current_mouse = Mouse.TOP
        logging.info(f"Current mouse: {current_mouse}")
        return current_mouse


    def get_mouse_current_position(self, observation, mouse):
        if mouse == self.current_mouse:
            mouse_factor = 1
        else:
            mouse_factor = -1
        current_map = mouse_factor*observation[:, :, 0]
        position = self.get_current_position(current_map)
        return position 


    def update_current_positions(self, observation, mouse):
        current_position = self.get_mouse_current_position(observation, mouse)
        column = current_position[1]
        if self.mouse_current_positions[mouse] is None \
            or self.mouse_current_positions[mouse].position != current_position:

            self.mouse_current_positions[mouse] = PositionCounter(current_position)
        else:
            self.mouse_current_positions[mouse].count += 1

        if self.mouse_current_columns[mouse] is None \
            or self.mouse_current_columns[mouse].column != column:

            self.mouse_current_columns[mouse] = ColumnCounter(column)
        else:
            self.mouse_current_columns[mouse].count += 1


    def determine_current_state(self, observation, action, info):
        self.current_mouse = self.determine_current_mouse(observation, action, info)
        self.update_current_positions(observation, self.current_mouse)
        current_states = [
            state
            for (state, conditions) in self.state_conditions.items()
            if all(condition(observation) for condition in conditions)
        ]
        logging.debug(f"Current candidate states: {current_states}")
        if len(current_states) != 1:
            return State.INVALID
        else:
            return current_states[0]


    def save_observation(self, observation, filename):
        observation_path = f"{self.save_dir}/{filename}.pickle"
        logging.debug(f"Writing observation to {observation_path}")
        with open(observation_path, 'wb') as file:
            pickle.dump(observation, file)


    def determine_oracle_action(self, observation, state):
        if self.current_mouse != self.oracle_policy[state]["mouse"]:
            action = 9
        else:
            target = self.oracle_policy[state]["target"]
            maze = self.construct_maze(observation, maze=self.maze)
            source = self.get_mouse_current_position(observation, self.current_mouse)
            oracle_direction, _ = find_shortest_path(maze, source, target)
            action = ACTION_DIRECTION_MAP.get(oracle_direction, 0)
            logging.debug(f"Oracle Direction: {oracle_direction}")
        logging.debug(f"Oracle Action: {action}")
        return action


    def populate_info(self, observation, info, state):
        info["state"] = state
        info["action"] = self.determine_oracle_action(observation, state)
        return info


    def step(self, action):
        logging.info(f"Action: {action}")

        observation, reward, terminated, truncated, info = self.env.step(action)
        logging.info(f"Info: {info}")

        current_state = self.determine_current_state(observation, action, info)

        if current_state == State.INVALID:
            logging.error(f"Encountered state {State.INVALID}")
            terminated = True

            self.save_observation(observation, filename="observation")
            self.save_observation(self.previous_observation, filename="previous_observation")

            try:
                for i, condition in enumerate(self.state_conditions[self.previous_state]):
                    if not condition(observation):
                        logging.error(f"Invalid condition for state {self.previous_state}: {i}")
            except Exception:
                pass
            raise Exception(f"State {current_state} encountered")
        elif current_state < self.previous_state:
            terminated = True

        if not self.passthrough:
            shaped_reward = self.shape_reward(observation, reward, action, current_state, 
                                              self.previous_state, terminated or truncated)
            terminated = terminated or truncated or current_state == State.WIN
        else:
            shaped_reward = reward

        self.cumulative_reward += shaped_reward

        if current_state != self.previous_state:
            self.previous_states.add(self.previous_state)
            self.previous_distance_to_target = {
                Mouse.TOP: None,
                Mouse.BOTTOM: None
            }

        info = self.populate_info(observation, info, current_state)

        self.previous_state = current_state
        self.previous_action = action
        self.previous_observation = observation
        self.previous_info = info

        logging.info(f"State: {current_state}")
        logging.info(f"Previous State: {self.previous_state}")
        logging.info(f"Previous states: {self.previous_states}")
        logging.info(f"Current reward: {shaped_reward}")
        logging.info(f"Cumulative reward: {self.cumulative_reward}")

        coordinate = ['Row', 'Column']
        mouse_positions = {
            f"{mouse} {coordinate[i]}": (
                self.mouse_current_positions[mouse].position[i]
                if self.mouse_current_positions[mouse] is not None else
                None
            )
            for mouse in [Mouse.TOP, Mouse.BOTTOM]
            for i in range(2)
        }
        self.wandb_log(mouse_positions)
        self.wandb_log({
            "Reward": shaped_reward,
            "Cumulative Reward": self.cumulative_reward
        })

        return observation, shaped_reward, terminated, truncated, info


    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        logging.info("Environment has been reset")
        observation, info = self.env.reset()
        self.previous_distance_to_target = None
        self.cumulative_reward = 0
        self.previous_distance_to_target = {
            Mouse.TOP: None,
            Mouse.BOTTOM: None
        }
        self.cumulative_directional_award = {
            Mouse.TOP: 0,
            Mouse.BOTTOM: 0
        }
        self.mouse_current_positions = {
            Mouse.TOP: None,
            Mouse.BOTTOM: None
        }
        self.mouse_current_columns = {
            mouse: None
            for mouse in [Mouse.TOP, Mouse.BOTTOM]
        }
        self.previous_state = None

        if self.previous_info is not None:
            self.previous_states = {State.RESET}
        else:
            self.previous_states = set()

        action = 0
        self.previous_state = self.determine_current_state(
            observation,
            action,
            info
        )
        self.previous_mouse = self.current_mouse
        self.previous_observation = observation
        self.previous_info = info
        return observation, info


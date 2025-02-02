import sys
import argparse
import numpy
import gymnasium as gym
import logging
import numpy as np

from arenax_sai import SAIClient
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EventCallback, CheckpointCallback

from solution import RewardShapingWrapper, get_key

ACTION_MAP = {
    'w': 3,
    'a': 7,
    's': 6,
    'd': 8,
    '9': 9
}

class CustomTrainingCallback(BaseCallback):
    """
    Custom callback for tracking training metrics and executing code during training.
    """
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        """
        Called after each step in the environment.
        Return False to stop training.
        """
        self.env.render()
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["interactive", "train"])
    parser.add_argument("--save_dir", type=str, default="./")
    parser.add_argument("--loglevel", type=str, choices=["INFO","DEBUG"], default='INFO')
    args = parser.parse_args()

    np.set_printoptions(threshold=sys.maxsize)

    sai = SAIClient(competition_id="U7SbGI4vFZdn")
    original_env = sai.make_env()

    # Configure logging with a specific format
    logging.basicConfig(
        level=args.loglevel,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    env = RewardShapingWrapper(original_env)

    if args.mode == "train":
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=args.save_dir,
            name_prefix="periodic_checkpoint"
        )
        model = PPO("MlpPolicy", env, verbose=0)
        training_callback = CustomTrainingCallback(env)
        model.learn(total_timesteps=1e15, callback=[training_callback, checkpoint_callback])
    else:
        env.reset()
        while True:
            raw_action = None
            while raw_action not in ACTION_MAP.keys():
                raw_action = get_key()
                logging.debug(f"Action: {raw_action}")
            action = ACTION_MAP[raw_action]
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated:
                env.reset()
            env.render()
            


if __name__ == "__main__":
    main()
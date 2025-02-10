import sys
import argparse
import numpy
import gymnasium as gym
import logging
import numpy as np

from arenax_sai import SAIClient
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, 
    EventCallback, 
    CheckpointCallback
)

from env import RewardShapingWrapper, get_key, State
from callbacks import RenderCallback, PretrainingCallback

ACTION_MAP = {
    'w': 3,
    'a': 7,
    's': 6,
    'd': 8,
    '9': 9,
    '0': 0,
    'q': 'q',
    't': 't'
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, 
                        choices=["interactive", "train", "demonstrate"])
    parser.add_argument("--save_dir", type=str, default="./")
    parser.add_argument("--loglevel", type=str, choices=["INFO","DEBUG"], 
                        default='INFO')
    parser.add_argument("--passthrough", action="store_true", default=False)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--log_file", type=str)
    parser.add_argument("--ent_coef", type=float, default=0.1)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--oracle_state", type=int, default=0,
                        choices=list(range(8)))
    parser.add_argument("--maze_path", type=str)
    parser.add_argument("--demos_path", type=str)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_exploration", action="store_true", default=False)
    parser.add_argument("--pretrain_save_location", type=str, default=None)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    args = parser.parse_args()

    np.set_printoptions(threshold=sys.maxsize)

    sai = SAIClient(competition_id="U7SbGI4vFZdn")
    original_env = sai.make_env()

    logging_kwargs = {
        "level": args.loglevel,
        "format": '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
    if args.log_file:
        logging_kwargs["filename"] = args.log_file
        logging_kwargs["filemode"] = 'w'

    # Configure logging with a specific format
    logging.basicConfig(**logging_kwargs)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    env_kwargs = {
        "env": original_env, 
        "passthrough": args.passthrough,
        "use_wandb": args.use_wandb,
        "maze_path": args.maze_path
    }

    env = RewardShapingWrapper(**env_kwargs)

    if args.mode == "train":
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=args.save_dir,
            name_prefix="periodic_checkpoint"
        )
        if args.checkpoint:
            model = PPO.load(args.checkpoint, env=env)
        else:
            model = PPO("MlpPolicy", env, verbose=0)

        model.set_parameters({
            'policy': {
                "ent_coef": args.ent_coef
            }
        }, exact_match=False)

        render_callback = RenderCallback(env)
        callbacks = [
            render_callback,
            checkpoint_callback
        ]

        if args.demos_path:
            demos = np.load(args.demos_path, allow_pickle=True)
            if args.batch_size == -1:
                batch_size = len(demos)
            else:
                batch_size = args.batch_size
            pretraining_callback = PretrainingCallback(
                demonstrations=demos,
                n_epochs=args.n_epochs,
                lr=args.lr,
                batch_size=batch_size,
                pretrain_save_location=args.pretrain_save_location
            )
            callbacks.append(pretraining_callback)

        model.learn(
            total_timesteps=1e15, 
            callback=callbacks
        )
    elif args.mode == "demonstrate":
        if args.checkpoint:
            model = PPO.load(args.checkpoint, env=env)
        else:
            model = PPO("MlpPolicy", env, verbose=0)

        demonstrations = []

        oracle_state = State(args.oracle_state)

        # Run the trained model
        obs, info = env.reset()
        try:
            while True:
                if (
                    args.oracle_state is not None \
                    and info.get("state", State.RESET) >= oracle_state
                ):

                    action = info["action"]
                else:
                    action, _ = model.predict(obs, deterministic=True)

                demonstration = {
                    "observation": obs,
                    "action": action
                }
                demonstrations.append(demonstration)
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                if terminated or truncated:
                    obs, info = env.reset()
        except KeyboardInterrupt:
            if args.demos_path:
                logging.info(f"Saving demonstrations to path {args.demos_path}")
                np.save(args.demos_path, demonstrations, allow_pickle=True)
    else:
        env.reset()
        while True:
            raw_action = None
            while raw_action not in ACTION_MAP.keys():
                raw_action = get_key()
                logging.debug(f"Action: {raw_action}")
            if raw_action == 'q':
                break
            elif raw_action == 't':
                env.reset()
            action = ACTION_MAP[raw_action]
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated:
                env.reset()
            env.render()
            


if __name__ == "__main__":
    main()
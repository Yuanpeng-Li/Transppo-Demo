import os
import sys
import time
import torch as th
import numpy as np
import random
from mt_transrl.policies import ACPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from mt_transrl.custom_network import Custom_FEN, Custom_Resnet
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box
from stable_baselines3.common.callbacks import BaseCallback
import metaworld


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, envs_name, verbose=0):
        super().__init__(verbose)
        self.env_name = envs_name

    def _on_step(self) -> bool:
        if self.locals["infos"][0]["TimeLimit.truncated"]:
            self.episode_ends[0] += 1
            self.episode_wins[0] += self.locals["infos"][0]["success"]
        return True

    def _on_rollout_start(self):
        self.episode_ends = [0] * 1
        self.episode_wins = [0] * 1

    def _on_rollout_end(self):
        self.logger.record(f"episode/{self.env_name}", self.episode_ends[0])
        self.logger.record(
            f"winning_rate/{self.env_name}", self.episode_wins[0] / self.episode_ends[0]
        )


NUM_ENV = 1
LOG_DIR = r"main/logs/MultiPPO"
os.makedirs(LOG_DIR, exist_ok=True)


# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert initial_value > 0.0

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler


def make_env(env_name, seed=0):
    def _init():
        env_cls = mt10.train_classes[env_name]
        env = env_cls()
        task = random.choice(
            [task for task in mt10.train_tasks if task.env_name == env_name]
        )
        env.set_task(task)
        env.seed(seed)
        env = Monitor(env)  # Use Monitor to record statistics
        return env

    return _init


class Combined_Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, features_dim: int = 512) -> None:
        super().__init__(observation_space, features_dim)
        self.Custom_FEN = Custom_FEN(observation_space, 512)
        PPO_observation_space = Box(
            -np.inf, np.inf, shape=(features_dim,), dtype=np.float32
        )
        self.Custom_Resnet = Custom_Resnet(PPO_observation_space, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.Custom_Resnet(self.Custom_FEN(observations))


mt10 = metaworld.MT10()


def main():
    names = list(mt10.train_classes.keys())[:5]
    for name in names:

        env = SubprocVecEnv([make_env(name) for i in range(NUM_ENV)])
        lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
        clip_range_schedule = linear_schedule(0.15, 0.025)

        policy_kwargs = dict(
            net_arch=dict(pi=[64, 32, 32], vf=[64, 32, 32, 32]),
        )
        model = PPO(
            ACPolicy,
            env,
            device="cuda",
            verbose=1,
            n_steps=int(1e4),
            batch_size=int(1e4),
            n_epochs=int(1e3),
            ent_coef=0.001,
            gamma=0.94,
            learning_rate=lr_schedule,
            clip_range=clip_range_schedule,
            tensorboard_log=LOG_DIR,
            policy_kwargs=policy_kwargs,
        )

        # Set the save directory
        save_dir = r"main/trained_models/MultiPPO/{0}".format(name)
        os.makedirs(save_dir, exist_ok=True)

        tensorboard_callback = TensorboardCallback(envs_name=name)
        # Writing the training logs from stdout to a file
        original_stdout = sys.stdout
        log_file_path = os.path.join(save_dir, "training_log.txt")
        with open(log_file_path, "w") as log_file:
            sys.stdout = log_file

            model.learn(
                total_timesteps=int(2500000),
                callback=[tensorboard_callback],
                tb_log_name=name,
            )
            env.close()

        # Restore stdout
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()

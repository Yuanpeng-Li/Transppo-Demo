import os
import sys
import time
import torch as th
import numpy as np
import random
from mt_transrl.single_ppo import single_PPO
from mt_transrl.policies import ACPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from mt_transrl.custom_network import Custom_FEN, Custom_Resnet
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium.spaces import Box
from stable_baselines3.common.callbacks import BaseCallback
import metaworld

mt = metaworld.MT10()
# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """    
    def __init__(self, envs_names, verbose=0):
        super().__init__(verbose)
        self.ep_rewards = [[] for _ in range(len(envs_names))]
        self.rewards = [0] * len(envs_names)
        self.env_names = envs_names 

    def _on_step(self) -> bool:
        self.update_ep_rewards()
        for polid in range(self.locals['env'].num_envs):
            if self.locals['infos'][polid]['TimeLimit.truncated']:
                self.episode_ends[polid] += 1
                self.episode_wins[polid] += self.locals['infos'][polid]['success']       
        return True
    
    def _on_rollout_start(self):
        self.episode_ends = [0] * 10
        self.episode_wins = [0] * 10
    
    def _on_rollout_end(self): 
        for i in range(self.locals['env'].num_envs):
            if self.ep_rewards[i] != []:
                env_name = self.env_names[i] 
                self.logger.record("rewards/{} (env #{})".format(env_name, i+1), np.mean(self.ep_rewards[i]))
        self.ep_rewards = [[] for _ in range(len(self.env_names))]
        self.rewards = [0] * len(self.env_names)
        for polid in range(self.locals['env'].num_envs):
            self.logger.record(f"episode/{self.env_names[polid]}", self.episode_ends[polid])
            self.logger.record(f"winning_rate/{self.env_names[polid]}", self.episode_wins[polid] / self.episode_ends[polid])
    
    def update_ep_rewards(self):
        for idx, info in enumerate(self.locals['infos']):
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                self.ep_rewards[idx].append(maybe_ep_info['r'])

    
class Combined_Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, features_dim: int = 512) -> None:
        super().__init__(observation_space, features_dim)
        self.Custom_FEN = Custom_FEN(observation_space, 512)
        processed_observation_space = Box(-np.inf, np.inf, shape = (features_dim,), dtype=np.float32)
        self.Custom_Resnet = Custom_Resnet(processed_observation_space, features_dim)
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.Custom_Resnet(self.Custom_FEN(observations))

def make_env(env_name, seed=0):
    def _init():
        env_cls = mt.train_classes[env_name]
        env = env_cls()
        task = random.choice([task for task in mt.train_tasks if task.env_name == env_name])
        env.set_task(task)
        env.seed(seed)
        env = Monitor(env)  # Use Monitor to record statistics
        return env
    return _init


NUM_ENV = 1
LOG_DIR = r'main/logs/SinglePPO'
save_dir = r"main/trained_models/SinglePPO"
os.makedirs(LOG_DIR, exist_ok=True)


def main():
    t1 = time.time()
    # Set up the environment and model
    envs = []
    envs_names = []
    for name in mt.train_classes.keys():
        envs.append(make_env(name))
        envs_names.append(name)
    env = SubprocVecEnv(envs)
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
    clip_range_schedule = linear_schedule(0.15, 0.025)
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 32, 32], vf=[64,32,32,32]),
    )
    model = single_PPO(
        ACPolicy,
        env,
        device="cuda",
        verbose=1,
        n_steps = int(1e4),
        batch_size = int(1e4),
        n_epochs = int(1e3),
        ent_coef = 0.001,
        gamma=0.94,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        tensorboard_log=LOG_DIR,
        policy_kwargs=policy_kwargs
    )
    # Set the save directory
    
    os.makedirs(save_dir, exist_ok=True)

    tensorboard_callback = TensorboardCallback(envs_names=envs_names)

    # Writing the training logs from stdout to a file
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file

        model.learn(
            total_timesteps=int(5e7),
            callback=[tensorboard_callback],
            tb_log_name = "SinglePPO"
        )
        env.close()

    # Restore stdout
    sys.stdout = original_stdout
    t2 = time.time()
    print("Training is done!")
    print("Time elapsed: {} seconds".format(t2-t1))

if __name__ == "__main__":
    main()



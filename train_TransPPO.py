import os
import sys
import numpy as np
from mt_transrl import Transppo_policy, Transppo, Custom_FEN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import metaworld, random

mt10 = metaworld.MT10()


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
        for polid in range(self.locals["env"].num_envs):
            if self.locals["infos"][polid]["TimeLimit.truncated"]:
                self.episode_ends[polid] += 1
                self.episode_wins[polid] += self.locals["infos"][polid]["success"]
        return True

    def _on_rollout_start(self):
        self.episode_ends = [0] * len(self.env_names)
        self.episode_wins = [0] * len(self.env_names)

    def _on_rollout_end(self):
        for i in range(self.locals["env"].num_envs):
            if self.ep_rewards[i] != []:
                env_name = self.env_names[i]
                self.logger.record(
                    "rewards/{} (env #{})".format(env_name, i + 1),
                    np.mean(self.ep_rewards[i]),
                )
        self.ep_rewards = [[] for _ in range(len(self.env_names))]
        self.rewards = [0] * len(self.env_names)
        for polid in range(self.locals["env"].num_envs):
            self.logger.record(
                f"episode/{self.env_names[polid]}", self.episode_ends[polid]
            )
            self.logger.record(
                f"winning_rate/{self.env_names[polid]}",
                self.episode_wins[polid] / self.episode_ends[polid],
            )

    def update_ep_rewards(self):
        for idx, info in enumerate(self.locals["infos"]):
            maybe_ep_info = info.get("episode")
            if maybe_ep_info is not None:
                self.ep_rewards[idx].append(maybe_ep_info["r"])


NUM_ENV = 1


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


def main():
    envs = []
    envs_names = []
    for name in mt10.train_classes.keys():
        envs.append(make_env(name))
        envs_names.append(name)
    env = SubprocVecEnv(envs)
    LOG_DIR = r"main/logs/TransPPO"

    print("Training on environment: {0}".format(envs_names))

    os.makedirs(LOG_DIR, exist_ok=True)

    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)

    clip_range_schedule = linear_schedule(0.15, 0.025)

    policy_kwargs = dict(
        features_extractor_class=Custom_FEN,
        net_arch=dict(pi=[32], vf=[32, 32]),
        share_features_extractor=True,
    )

    model = Transppo(
        Transppo_policy,
        env,
        device="cuda",
        verbose=1,
        env_names=envs_names,
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
    save_dir = r"main/trained_models/transPPO"

    tensorboard_callback = TensorboardCallback(envs_names=envs_names)
    # Writing the training logs from stdout to a file
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, "w") as log_file:
        sys.stdout = log_file

        model.learn(total_timesteps=int(5e7), callback=[tensorboard_callback])
        env.close()

    # Restore stdout
    sys.stdout = original_stdout


if __name__ == "__main__":
    main()

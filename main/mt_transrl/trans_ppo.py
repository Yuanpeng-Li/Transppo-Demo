import warnings
import sys
import time
from typing import (
    Any,
    ClassVar,
    Dict,
    Optional,
    Type,
    TypeVar,
    Union,
    Generator,
    NamedTuple,
)
import copy
import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.buffers import BaseBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import (
    explained_variance,
    get_schedule_fn,
    get_device,
    obs_as_tensor,
    safe_mean,
)

from torchjd import mtl_backward
from torchjd.aggregation import UPGrad

SelfTransppoAlgorithm = TypeVar("SelfTransppoAlgorithm", bound="Transppo")


class Transppo_RolloutBufferSamples(NamedTuple):
    observations: th.Tensor  # Keeps (batch_size, n_envs, *obs_shape)
    actions: th.Tensor  # Keeps (batch_size, n_envs, action_dim)
    old_values: th.Tensor  # Keeps (batch_size, n_envs)
    old_log_prob: th.Tensor  # Keeps (batch_size, n_envs)
    advantages: th.Tensor  # Keeps (batch_size, n_envs)
    returns: th.Tensor  # Keeps (batch_size, n_envs)


class Transppo_RolloutBuffer(RolloutBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            gae_lambda,
            gamma,
            n_envs=n_envs,
        )

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[Transppo_RolloutBufferSamples, None, None]:
        """
        Custom `get` method that keeps the env dim (buffer_size, n_envs, ...).
        """
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size)
        # Prepare the data
        if not self.generator_ready:
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size

        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray) -> "Transppo_RolloutBufferSamples":
        """
        Custom `_get_samples` method that keeps the env dim (buffer_size, n_envs, ...).
        """
        data = (
            self.observations[batch_inds],  # Keeps (batch_size, n_envs, *obs_shape)
            self.actions[batch_inds],  # Keeps (batch_size, n_envs, action_dim)
            self.values[batch_inds],  # Keeps (batch_size, n_envs)
            self.log_probs[batch_inds],  # Keeps (batch_size, n_envs)
            self.advantages[batch_inds],  # Keeps (batch_size, n_envs)
            self.returns[batch_inds],  # Keeps (batch_size, n_envs)
        )
        return Transppo_RolloutBufferSamples(*tuple(map(self.to_torch, data)))


class Transppo(OnPolicyAlgorithm):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        env_names: Optional[list] = None,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = Transppo_RolloutBuffer,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            # buffer_size = self.env.num_envs * self.n_steps
            buffer_size = self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.env_names = env_names

        if self.policy_kwargs == None:
            self.policy_kwargs = dict(n_envs=self.n_envs)
        else:
            self.policy_kwargs["n_envs"] = self.n_envs

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, (
                    "`clip_range_vf` must be positive, "
                    "pass `None` to deactivate vf clipping"
                )

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)
        self.policy_cpu = copy.deepcopy(self.policy).to("cpu")

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy_cpu.load_state_dict(self.policy.state_dict())
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy_cpu.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy_cpu.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy_cpu.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, th.device("cpu"))
                actions, values, log_probs = self.policy_cpu(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy_cpu.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy_cpu.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += 1

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLismit.truncated", False)
                ):
                    terminal_obs = self.policy_cpu.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy_cpu.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy_cpu.predict_values(obs_as_tensor(new_obs, th.device("cpu")))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        clip_fractions_batches = []
        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            approx_kl_divs_batches = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = (
                        rollout_data.actions.long().flatten()
                    )  # actions: (batch_size, n_envs) observations: (batch_size, n_envs, *obs_shape)

                values_list, log_prob_list, entropy_list, features = (
                    self.policy.evaluate_actions(rollout_data.observations, actions)
                )  # values: (batch_size, n_envs, 1) log_prob: (batch_size, n_envs) entropy: (batch_size, n_envs)

                # Normalize advantage
                advantages = rollout_data.advantages  # advantages: (batch_size, n_envs)
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # ratio between old and new policy, should be one at the first iteration
                ratio_list = [
                    th.exp(log_prob - rollout_data.old_log_prob[:, i])
                    for i, log_prob in enumerate(log_prob_list)
                ]  # ratio: (batch_size, n_envs)

                # clipped surrogate loss
                policy_loss_1 = [
                    advantages[:, i] * ratio for i, ratio in enumerate(ratio_list)
                ]  # policy_loss_1: (batch_size, n_envs)
                policy_loss_2 = [
                    advantages[:, i] * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    for i, ratio in enumerate(ratio_list)
                ]
                policy_loss = [
                    -th.min(policy_loss_1[i], policy_loss_2[i]).mean()
                    for i in range(len(policy_loss_1))
                ]  # policy_loss: (n_envs)

                # Logging
                for loss in policy_loss:
                    pg_losses.append(loss.item())

                clip_fraction = th.mean(
                    (th.abs(th.stack(ratio_list, dim=1) - 1) > clip_range).float()
                )

                clip_fractions.append(clip_fraction.item())
                # --- Per-Environment Clip Fraction Calculation ---
                clip_fractions_per_env = []
                for i in range(self.n_envs):
                    # Calculate fraction where prediction deviates significantly for this env
                    fraction = th.mean(
                        (th.abs(ratio_list[i] - 1) > clip_range).float()
                    ).item()
                    clip_fractions_per_env.append(fraction)
                # Store list of per-env fractions for this batch
                clip_fractions_batches.append(clip_fractions_per_env)
                # -------------------------------------------------

                # --- Per-Environment KL Divergence Calculation ---
                approx_kl_divs_per_env = []
                with th.no_grad():
                    for i in range(self.n_envs):
                        log_ratio_env = (
                            log_prob_list[i] - rollout_data.old_log_prob[:, i]
                        )  # (batch_size,)
                        # Ensure log_ratio_env is on CPU before converting to numpy
                        kl_div_env = (
                            th.mean((th.exp(log_ratio_env) - 1) - log_ratio_env)
                            .cpu()
                            .numpy()
                        )
                        approx_kl_divs_per_env.append(
                            kl_div_env.item()
                        )  # Use .item() to get scalar
                # Store list of per-env KLs for this batch
                approx_kl_divs_batches.append(approx_kl_divs_per_env)
                # --------------------------------------------------

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values_list
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = [
                        rollout_data.old_values[:, i]
                        + th.clamp(
                            values - rollout_data.old_values[:, i],
                            -clip_range_vf,
                            clip_range_vf,
                        )
                        for i, values in enumerate(values_list)
                    ]
                # Value loss using the TD(gae_lambda) target
                value_loss = [
                    F.mse_loss(rollout_data.returns[:, i].unsqueeze(-1), values_pred[i])
                    for i in range(self.n_envs)
                ]

                value_losses.append(th.mean(th.stack(value_loss)).item())

                # Entropy loss favor exploration
                if entropy_list is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = [-th.mean(-log_prob) for log_prob in log_prob_list]
                else:
                    entropy_loss = -th.mean(log_prob_list, dim=0)
                if self.ent_coef > 0:
                    entropy_loss = [
                        th.clamp(entropy_loss[i], -5, 5) for i in range(self.n_envs)
                    ]
                entropy_losses.append(th.mean(th.stack(entropy_loss)).item())
                # --- Per-Environment Policy Coefficient Calculation ---
                # Determine policy coefficient based on individual environment's clip fraction
                policy_coefs_per_env = []
                for approx_kl in approx_kl_divs_per_env:
                    if approx_kl > 0.03:
                        policy_coefs_per_env.append(
                            0
                        )  # Reduce weight if policy changed drastically
                    elif approx_kl > 0.02:
                        policy_coefs_per_env.append(0.5)
                    elif approx_kl > 0.01:
                        policy_coefs_per_env.append(0.75)
                    else:
                        policy_coefs_per_env.append(1.0)  # Keep full weight otherwise
                # -----------------------------------------------------

                all_loss = [
                    policy_coef * policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    for policy_coef, policy_loss, entropy_loss, value_loss in zip(
                        policy_coefs_per_env, policy_loss, entropy_loss, value_loss
                    )
                ]
                # loss = all_loss.mean()
                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = (
                        th.stack(log_prob_list, dim=1) - rollout_data.old_log_prob
                    )
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()

                mtl_backward(
                    losses=all_loss,
                    features=features,
                    aggregator=UPGrad(),
                    retain_graph=True,
                )
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            log_std_tensor = th.cat([p.view(-1) for p in self.policy.log_std])
            self.logger.record("train/std", th.exp(log_std_tensor).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        # --- Log Per-Environment and Overall KL and Clip Fraction ---
        if approx_kl_divs_batches:  # Check if list is not empty
            mean_kls_per_env = np.mean(
                np.array(approx_kl_divs_batches), axis=0
            )  # Shape: (n_envs,)
            for i in range(self.n_envs):
                self.logger.record(
                    f"approx_kl/{self.env_names[i]}", mean_kls_per_env[i]
                )

        if clip_fractions_batches:  # Check if list is not empty
            mean_clips_per_env = np.mean(
                np.array(clip_fractions_batches), axis=0
            )  # Shape: (n_envs,)
            for i in range(self.n_envs):
                self.logger.record(
                    f"clip_fraction/{self.env_names[i]}", mean_clips_per_env[i]
                )
        # -------------------------------------------------------------

    def learn(
        self: SelfTransppoAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "TransPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTransppoAlgorithm:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max(
            (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon
        )
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record(
                "rollout/ep_rew_mean",
                safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
            )
            self.logger.record(
                "rollout/ep_len_mean",
                safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
            )
        self.logger.record("time/fps", fps)
        self.logger.record(
            "time/time_elapsed", int(time_elapsed), exclude="tensorboard"
        )
        self.logger.record(
            "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
        )
        if len(self.ep_success_buffer) > 0:
            self.logger.record(
                "rollout/success_rate", safe_mean(self.ep_success_buffer)
            )
        self.logger.dump(step=self.num_timesteps)

    def _update_learning_rate(
        self, optimizers: Union[list[th.optim.Optimizer], th.optim.Optimizer]
    ) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a list of optimizers.
        """
        # Log the current learning rate
        self.logger.record(
            "train/learning_rate", self.lr_schedule(self._current_progress_remaining)
        )

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            optimizer.param_groups[0]["lr"] = (
                self.lr_schedule(self._current_progress_remaining) / self.n_envs
            )
            optimizer.param_groups[1]["lr"] = self.lr_schedule(
                self._current_progress_remaining
            )

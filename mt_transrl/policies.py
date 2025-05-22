import collections
import copy
import os
import random
import sys
import time
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gymnasium as spaces
import metaworld
import numpy as np
import torch as th
from gymnasium.spaces import Box
from mt_transrl.custom_network import Custom_FEN, Custom_Resnet, Custom_Split

# from street_fighter_custom_wrapper import StreetFighterCustomWrapper
from mt_transrl.single_ppo import single_PPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    SquashedDiagGaussianDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.preprocessing import (
    get_action_dim,
    is_image_space,
    maybe_transpose,
    preprocess_obs,
)
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.utils import (
    get_device,
    is_vectorized_observation,
    obs_as_tensor,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch import nn


class ACPolicy(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.`
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )

        if (
            isinstance(net_arch, list)
            and len(net_arch) > 0
            and isinstance(net_arch[0], dict)
        ):
            warnings.warn(
                (
                    "As shared layers in the mlp_extractor are removed since SB3 v1.8.0, "
                    "you should now pass directly a dictionary and not a list "
                    "(net_arch=dict(pi=..., vf=...) instead of net_arch=[dict(pi=..., vf=...)])"
                ),
            )
            net_arch = net_arch[0]

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = dict(pi=[64, 64], vf=[64, 64])

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.share_features_extractor = share_features_extractor
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        if self.share_features_extractor:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
        else:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.make_features_extractor()

        self.log_std_init = log_std_init
        dist_kwargs = None

        assert not (
            squash_output and not use_sde
        ), "squash_output=True is only available when using gSDE (use_sde=True)"
        # Keyword arguments for gSDE distribution
        if use_sde:
            dist_kwargs = {
                "full_std": full_std,
                "squash_output": squash_output,
                "use_expln": use_expln,
                "learn_features": False,
            }

        self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # Action distribution
        self.action_dist = SquashedDiagGaussianDistribution(
            get_action_dim(action_space)
        )

        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)  # type: ignore[arg-type, return-value]

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        assert isinstance(
            self.action_dist, StateDependentNoiseDistribution
        ), "reset_noise() is only available when using gSDE"
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi,
                latent_sde_dim=latent_dim_pi,
                log_std_init=self.log_std_init,
            )
        elif isinstance(
            self.action_dist,
            (
                CategoricalDistribution,
                MultiCategoricalDistribution,
                BernoulliDistribution,
            ),
        ):
            self.action_net = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi
            )
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def extract_features(  # type: ignore[override]
        self,
        obs: PyTorchObs,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> Union[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """
        Preprocess the observation if needed and extract features.

        :param obs: Observation
        :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
        :return: The extracted features. If features extractor is not shared, returns a tuple with the
            features for the actor and the features for the critic.
        """
        if self.share_features_extractor:
            return super().extract_features(
                obs,
                (
                    self.features_extractor
                    if features_extractor is None
                    else features_extractor
                ),
            )
        else:
            if features_extractor is not None:
                warnings.warn(
                    "Provided features_extractor will be ignored because the features extractor is not shared.",
                    UserWarning,
                )

            pi_features = super().extract_features(obs, self.pi_features_extractor)
            vf_features = super().extract_features(obs, self.vf_features_extractor)
            return pi_features, vf_features

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(
                mean_actions, self.log_std, latent_pi
            )
        else:
            raise ValueError("Invalid action distribution")

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation).get_actions(
            deterministic=deterministic
        )

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # # Disable orthogonal initialization
        # kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = Custom_Split(
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )


class Transppo_policy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        n_envs: int,
        *args,
        **kwargs,
    ):
        # # Disable orthogonal initialization
        # kwargs["ortho_init"] = False
        self.n_envs = n_envs
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def reset_noise(self, n_envs: int = 1) -> None:
        pass

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = nn.ModuleList(
            [
                Custom_Split(
                    net_arch=self.net_arch,
                    activation_fn=self.activation_fn,
                    device=self.device,
                )
                for _ in range(self.n_envs)
            ]
        )

    def _build(self, lr_schedule):
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self.action_dist = [
            SquashedDiagGaussianDistribution(get_action_dim(self.action_space))
            for _ in range(self.n_envs)
        ]

        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor[0].latent_dim_pi

        if isinstance(self.action_dist[0], DiagGaussianDistribution):
            self.action_net = nn.ModuleList(
                [
                    self.action_dist[i].proba_distribution_net(
                        latent_dim=latent_dim_pi, log_std_init=self.log_std_init
                    )[0]
                    for i in range(self.n_envs)
                ]
            )
            self.log_std = nn.ParameterList(
                [
                    nn.Parameter(
                        th.ones(self.action_dist[i].action_dim) * self.log_std_init,
                        requires_grad=True,
                    )
                    for i in range(self.n_envs)
                ]
            )

        elif isinstance(self.action_dist[0], StateDependentNoiseDistribution):
            self.action_net = nn.ModuleList(
                [
                    self.action_dist[i].proba_distribution_net(
                        latent_dim=latent_dim_pi,
                        latent_sde_dim=latent_dim_pi,
                        log_std_init=self.log_std_init,
                    )[0]
                    for i in range(self.n_envs)
                ]
            )
            self.log_std = nn.ParameterList(
                [
                    nn.Parameter(
                        th.ones(self.action_dist[i].action_dim) * self.log_std_init,
                        requires_grad=True,
                    )
                    for i in range(self.n_envs)
                ]
            )

        elif isinstance(
            self.action_dist[0],
            (
                CategoricalDistribution,
                MultiCategoricalDistribution,
                BernoulliDistribution,
            ),
        ):
            self.action_net = nn.ModuleList(
                [
                    self.action_dist[i].proba_distribution_net(latent_dim=latent_dim_pi)
                    for i in range(self.n_envs)
                ]
            )

        else:
            raise NotImplementedError(
                f"Unsupported distribution '{self.action_dist[0]}'."
            )

        self.value_net = nn.ModuleList(
            [
                nn.Linear(self.mlp_extractor[0].latent_dim_vf, 1)
                for _ in range(self.n_envs)
            ]
        )
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            self.features_extractor.apply(partial(self.init_weights, gain=np.sqrt(2)))
            for extractor in self.mlp_extractor:
                extractor.apply(partial(self.init_weights, gain=np.sqrt(2)))

            for module in self.action_net:
                module.apply(partial(self.init_weights, gain=0.01))

            for module in self.value_net:
                module.apply(partial(self.init_weights, gain=1))

        # Setup optimizer with initial learning rate
        # self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]
        self.optimizer = self.optimizer_class(
            [
                {
                    "params": list(self.features_extractor.parameters()),
                    "lr": lr_schedule(1) / self.n_envs,
                },
                {
                    "params": [
                        p
                        for name, p in self.named_parameters()
                        if "features_extractor" not in name
                    ],
                    "lr": lr_schedule(1),
                },
            ]
        )

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            # If feature extractor is shared, use the first one for all environments
            latent_pi_list, latent_vf_list = zip(
                *[
                    self.mlp_extractor[i](features[i].unsqueeze(0))
                    for i in range(self.n_envs)
                ]
            )
        else:
            # If not shared, split features and use separate extractors
            pi_features, vf_features = features
            latent_pi_list = [
                self.mlp_extractor[i].forward_actor(pi_features[i].unsqueeze(0))
                for i in range(self.n_envs)
            ]
            latent_vf_list = [
                self.mlp_extractor[i].forward_critic(vf_features[i].unsqueeze(0))
                for i in range(self.n_envs)
            ]

        # Evaluate the values for the given observations using `ModuleList`
        values_list = [self.value_net[i](latent_vf_list[i]) for i in range(self.n_envs)]

        distribution_list = [
            self._get_action_dist_from_latent(latent_pi_list[i], env_id=i)
            for i in range(self.n_envs)
        ]
        actions_list = [
            distribution_list[i].get_actions(deterministic=deterministic)
            for i in range(self.n_envs)
        ]
        log_prob_list = [
            distribution_list[i].log_prob(actions)
            for i, actions in enumerate(actions_list)
        ]
        actions_list = [
            actions_list[i].reshape((-1, *self.action_space.shape))
            for i in range(self.n_envs)
        ]
        actions = th.cat(actions_list, dim=0)  # Shape: (n_envs, action_dim)
        values = th.cat(values_list, dim=0)  # Shape: (n_envs, 1)
        log_prob = th.cat(log_prob_list, dim=0)  # Shape: (n_envs)
        return actions, values, log_prob

    def _get_action_dist_from_latent(
        self, latent_pi: th.Tensor, env_id: int
    ) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net[env_id](latent_pi)

        if isinstance(self.action_dist[env_id], DiagGaussianDistribution):
            return self.action_dist[env_id].proba_distribution(
                mean_actions, self.log_std[env_id]
            )
        elif isinstance(self.action_dist[env_id], CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist[env_id].proba_distribution(
                action_logits=mean_actions
            )
        elif isinstance(self.action_dist[env_id], MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist[env_id].proba_distribution(
                action_logits=mean_actions
            )
        elif isinstance(self.action_dist[env_id], BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist[env_id].proba_distribution(
                action_logits=mean_actions
            )
        elif isinstance(self.action_dist[env_id], StateDependentNoiseDistribution):
            return self.action_dist[env_id].proba_distribution(
                mean_actions, self.log_std[env_id], latent_pi
            )
        else:
            raise ValueError("Invalid action distribution")

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        pass  ##Not used in training ppo

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation shape:(batch_size, n_envs, *obs_shape)
        :param actions: Actions shape:(batch_size, n_envs, *action_shape)
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        batch_size, n_envs = obs.shape[
            :2
        ]  # Extract batch size and number of environments
        flattened_obs = obs.view(
            batch_size * n_envs, *obs.shape[2:]
        )  # Merge n_envs into batch

        # Pass through the feature extractor
        flattened_features = self.extract_features(
            flattened_obs
        )  # (batch_size * n_envs, *latent_feature)

        # Reshape it back to (batch_size, n_envs, *latent_feature)
        features = flattened_features.view(
            batch_size, n_envs, *flattened_features.shape[1:]
        )

        if self.share_features_extractor:
            # If feature extractor is shared, use the first one for all environments
            latent_pi_list, latent_vf_list = zip(
                *[self.mlp_extractor[i](features[:, i, :]) for i in range(self.n_envs)]
            )
        else:
            # If not shared, split features and use separate extractors
            pi_features, vf_features = features
            latent_pi_list = [
                self.mlp_extractor[i].forward_actor(pi_features[:, i])
                for i in range(self.n_envs)
            ]
            latent_vf_list = [
                self.mlp_extractor[i].forward_critic(vf_features[:, i])
                for i in range(self.n_envs)
            ]
        # Stack latent representations to keep batch processing format
        # latent_pi = th.stack(latent_pi_list, dim=1)  # Shape: (batch_size, n_envs, latent_dim_pi)
        # latent_vf = th.stack(latent_vf_list, dim=1)  # Shape: (batch_size, n_envs, latent_dim_vf)
        distribution_list = [
            self._get_action_dist_from_latent(latent_pi_list[i], env_id=i)
            for i in range(self.n_envs)
        ]
        values_list = [self.value_net[i](latent_vf_list[i]) for i in range(self.n_envs)]
        entropy_list = [distribution_list[i].entropy() for i in range(self.n_envs)]
        log_prob_list = [
            distribution_list[i].log_prob(actions[:, i]) for i in range(self.n_envs)
        ]
        # values = th.stack(values_list, dim=1)
        # log_prob = th.stack([distribution_list[i].log_prob(actions[:,i]) for i in range(self.n_envs)], dim=1)
        if None in entropy_list:
            entropy = None
        else:
            entropy = entropy_list
        return (
            values_list,
            log_prob_list,
            entropy,
            features,
        )  # values: [(batch_size, 1)], log_prob: [(batch_size)], entropy: [(batch_size)]

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation shape:(n_envs, *obs_shape)
        :return: the estimated values.
        """
        features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = th.cat(
            [
                self.mlp_extractor[i].forward_critic(features[i].unsqueeze(0))
                for i in range(self.n_envs)
            ],
            dim=0,
        )
        predict_values = th.cat(
            [self.value_net[i](latent_vf[i].unsqueeze(0)) for i in range(self.n_envs)],
            dim=0,
        )

        return predict_values

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.

        :param n_envs:
        """
        for i in range(self.n_envs):
            assert isinstance(
                self.action_dist[i], StateDependentNoiseDistribution
            ), "reset_noise() is only available when using gSDE"
            self.action_dist[i].sample_weights(self.log_std, batch_size=n_envs)

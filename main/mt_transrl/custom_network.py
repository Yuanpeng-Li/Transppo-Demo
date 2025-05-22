import torch as th
import gymnasium as gym
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from typing import Union, List, Dict, Type, Tuple
from stable_baselines3.common.utils import get_device


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Custom_FEN(BaseFeaturesExtractor):
    def __init__(
        self, observation_space, action_features_dim=32, value_features_dim=32
    ):

        super().__init__(observation_space, action_features_dim)

        self.flatten = nn.Flatten()

        input_dim = observation_space.shape[0]

        self.mlp_action = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, action_features_dim)
        )

        self.mlp_value = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, value_features_dim)
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.flatten(observations)
        action_output = self.mlp_action(x)
        value_output = self.mlp_value(x)
        return th.cat([action_output, value_output], dim=1)


class Custom_Resnet(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64) -> None:
        super().__init__(observation_space, features_dim)

        self.linear1 = nn.Linear(512, 128)
        self.linear2 = nn.Linear(128, features_dim)
        self.residual = nn.Linear(512, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        temp = self.linear1(observations)
        temp = nn.ReLU()(temp)
        temp = self.linear2(temp)
        temp = nn.ReLU()(temp)
        return nn.ReLU()(temp + self.residual(observations))


class Custom_Split(nn.Module):

    def __init__(
        self,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        action_dim: int = 32,
        value_dim: int = 32,
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        last_layer_dim_pi = action_dim
        last_layer_dim_vf = value_dim

        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specificed, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        # Iterate through the policy layers and build the policy net
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        # Iterate through the value layers and build the value net
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def split_combined_output(self, combined_output, action_dim, value_dim):
        """
        Split the combined output into action_output and value_output.

        :param combined_output: The concatenated tensor with action and value outputs.
        :param action_dim: The dimensionality of the action_output (default=32).
        :param value_dim: The dimensionality of the value_output (default=32).
        :return: action_output, value_output
        """
        # Extract action_output based on action_dim
        action_output = combined_output[:, :action_dim]

        # Extract value_output based on value_dim
        value_output = combined_output[:, action_dim : action_dim + value_dim]

        return action_output, value_output

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        action_output, _ = self.split_combined_output(
            features, self.latent_dim_pi, self.latent_dim_vf
        )
        return self.policy_net(action_output)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        _, value_output = self.split_combined_output(
            features, self.latent_dim_pi, self.latent_dim_vf
        )
        return self.value_net(value_output)


class function_f(nn.Module):
    def __init__(self, observation_space):
        super().__init__()

        # # Enhanced CNN architecture with deeper layers and residual connections
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 32, kernel_size=3, stride=2),
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        # )

        # # Compute shape by doing one forward pass
        # with th.no_grad():
        #     n_flatten = self.compute_flatten_size(observation_space)

        # Enhanced MLP with added Dropout for regularization
        self.mlp = nn.Sequential(
            nn.Linear(*observation_space.shape, 4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4, 1),
        )

    # def compute_flatten_size(self, observation_space):
    #     input = th.rand(1, *observation_space.shape)
    #     if input.permute(0, 1, 3, 2).shape[1] == 3:
    #         return self.cnn(input.permute(0, 1, 3, 2)).float().shape[1]
    #     else:
    #         return self.cnn(input.permute(0, 3, 1, 2)).float().shape[1]

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.mlp(observations)

import copy
import numpy as np
import torch

import torch.nn as nn
import torch.nn.init as init

# Define the weight initialization function
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.uniform_(m.weight, a=-1.0, b=1.0)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight, a=-1.0, b=1.0)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class CNN(nn.Module):

    def __init__(
            self,
            in_channels: int,
            conv_layers: int = 3,
            channels: list = [32, 64, 64],
            kernel_sizes: list = [8, 4, 3],
            strides: list = [4, 2, 1]):
        super().__init__()

        if not (len(channels) == conv_layers and len(kernel_sizes) == conv_layers and len(strides) == conv_layers):
            raise ValueError("Length of channels, kernel_sizes, and strides must match conv_layers")

        # Convolutional layers
        layers = [
            nn.Conv2d(in_channels, channels[0], kernel_sizes[0], strides[0]),
            nn.ReLU()
        ]
        for i in range(1, conv_layers):
            layers.append(nn.Conv2d(channels[i-1], channels[i], kernel_sizes[i], strides[i]))
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self.net.apply(weights_init)

    def copy(self):
        return copy.deepcopy(self)

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            hidden_layers: int = 2,
            hidden_units: list = [128, 128]):
        super().__init__()

        if len(hidden_units) != hidden_layers:
            raise ValueError("Length of hidden_units must match hidden_layers")

        layers = []
        layers.append(nn.Linear(in_features, hidden_units[0]))
        layers.append(nn.ReLU())

        for i in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_units[-1], out_features))
        self.net = nn.Sequential(*layers)
        self.net.apply(weights_init)

    def copy(self):
        return copy.deepcopy(self)

    def forward(self, x):
        return self.net(x)


class DQN(nn.Module):
    def __init__(
            self,
            cnn: nn.Module,
            mlp: nn.Module):
        super().__init__()

        self.net = nn.Sequential(
            cnn,
            nn.Flatten(),
            mlp
        )
        self.net.apply(weights_init)

    def copy(self):
        return copy.deepcopy(self)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    net1 = MLP(4, 2)
    state_sample = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    state_tensor = torch.tensor(state_sample)
    q_values = net1(state_tensor)
    assert isinstance(q_values, torch.Tensor)
    assert q_values.shape == (2,)

    net2 = CNN(1)

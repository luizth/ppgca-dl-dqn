from typing import Protocol

import numpy as np
import torch


class FeatureConstructor(Protocol):
    @staticmethod
    def encode(state, state_dim) -> dict:
        """Encode the given state into a feature representation."""
        pass


class DiscreteOneHot:
    @staticmethod
    def encode(state: int, state_dim: int) -> torch.Tensor:
        """Encode a discrete state as a one-hot vector."""
        one_hot = np.zeros(state_dim, dtype=np.float32)
        one_hot[state] = 1.
        return torch.tensor(one_hot)


class ContinuousNormalized:
    @staticmethod
    def encode(state: np.ndarray, state_dim: int) -> torch.Tensor:
        """Encode a continuous state as a normalized vector."""
        normalized = (state - np.min(state)) / (np.max(state) - np.min(state))
        return torch.tensor(normalized, dtype=torch.float32)


if __name__ == "__main__":
    # Example usage of DiscreteOneHot
    state = 2
    state_dim = 4
    one_hot_vector = DiscreteOneHot.encode(state, state_dim)
    assert isinstance(one_hot_vector, torch.Tensor)
    assert one_hot_vector.shape == (state_dim,)
    assert one_hot_vector.sum().item() == 1.0

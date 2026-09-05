from typing import Protocol

import gymnasium as gym
import numpy as np
import torch


class FeatureConstructor(Protocol):
    """A protocol for feature constructors."""

    def encode(self, state, state_dim) -> dict:
        """Encode the given state into a feature representation."""
        pass


class DiscreteOneHot:
    def __init__(self, state_space: gym.spaces.Discrete):
        self.state_dim = int(state_space.n)

    def encode(self, state: int) -> torch.Tensor:
        """Encode a discrete state as a one-hot vector."""
        one_hot = np.zeros(self.state_dim, dtype=np.float32)
        one_hot[int(state)] = 1.
        return torch.from_numpy(one_hot)


class ContinuousNormalized:
    """Normaliza CADA dimensão para [-1, 1] usando os limites do observation_space.

    Dimensões sem limite finito (as velocidades do CartPole) passam sem alteração,
    apenas com um clip de segurança. O clip também protege ambientes que estouram
    os próprios limites declarados — o LunarLander avisa disso na fonte.

    MountainCar-v0 — low = [-1.2, -0.07], high = [0.6, 0.07], ambas as dimensões limitadas:

    center = (high + low) / 2 = [(0.6 + (-1.2))/2, (0.07 + (-0.07))/2] = [-0.3,  0.00]
    scale  = (high - low) / 2 = [(0.6 - (-1.2))/2, (0.07 - (-0.07))/2] = [ 0.9,  0.07]

    center e scale são calculados uma vez no construtor. Depois, cada observação é só
    uma subtração e uma divisão, dimensão a dimensão:

    ┌─────────────────────────┬───────────────────────────────────┬─────────────────┬──────────────────────┐
    │ s (posição, velocidade) │               conta               │      novo       │   antigo (pre-fix)   │
    ├─────────────────────────┼───────────────────────────────────┼─────────────────┼──────────────────────┤
    │ [-0.50, 0.010]          │ [(-0.50+0.3)/0.9, (0.010-0)/0.07] │ [-0.222, 0.143] │ [0.0, 1.0]           │
    ├─────────────────────────┼───────────────────────────────────┼─────────────────┼──────────────────────┤
    │ [-0.50, 0.020]          │ [(-0.50+0.3)/0.9, (0.020-0)/0.07] │ [-0.222, 0.286] │ [0.0, 1.0]           │
    ├─────────────────────────┼───────────────────────────────────┼─────────────────┼──────────────────────┤
    │ [-0.40, 0.010]          │ [(-0.40+0.3)/0.9, (0.010-0)/0.07] │ [-0.111, 0.143] │ [0.0, 1.0]           │
    └─────────────────────────┴───────────────────────────────────┴─────────────────┴──────────────────────┘

    Três estados fisicamente distintos — velocidade dobrada na segunda linha, posição 0.1 adiante na terceira
    — colapsam no mesmo vetor [0.0, 1.0] no encoder antigo, porque com 2 dimensões o min-max sempre manda a menor
    para 0 e a maior para 1. Cada dimensão é medida contra a própria escala física e os três saem diferentes.

    CartPole-v1 — o caso misto, low = [-4.8, -inf, -0.419, -inf]:

    bounded = [True, False, True, False]
    center  = [0.0, 0.0, 0.0,   0.0]
    scale   = [4.8, 1.0, 0.419, 1.0]      <- 1.0 nas duas velocidades (sem limite finito)

    Para s = [-0.0414, -0.0263, 0.0301, 0.0082]:

            posição:  (-0.0414 - 0) / 4.8    = -0.00862
        vel. linear:    (-0.0263 - 0) / 1.0    = -0.02630   <- passa direto, só o clip
            ângulo:   ( 0.0301 - 0) / 0.419  =  0.07186
        vel. angular:   ( 0.0082 - 0) / 1.0    =  0.00820   <- passa direto

    NOVO   = [-0.009, -0.026,  0.072,  0.008]
    ANTIGO = [ 0.000,  0.211,  1.000,  0.694]
    """
    def __init__(self, state_space: gym.spaces.Box, clip: float = 10.0):
        low = np.asarray(state_space.low, dtype=np.float64)
        high = np.asarray(state_space.high, dtype=np.float64)
        bounded = np.isfinite(low) & np.isfinite(high)

        center = np.where(bounded, (high + low) / 2.0, 0.0)
        scale = np.where(bounded, (high - low) / 2.0, 1.0)
        scale[scale == 0.0] = 1.0  # dimensão degenerada (low == high)

        self.state_dim = int(state_space.shape[0])
        self.center = center.astype(np.float32)
        self.scale = scale.astype(np.float32)
        self.clip = clip

    def encode(self, state: np.ndarray) -> torch.Tensor:
        x = (np.asarray(state, dtype=np.float32) - self.center) / self.scale
        return torch.from_numpy(np.clip(x, -self.clip, self.clip))


if __name__ == "__main__":
    # Example usage of DiscreteOneHot
    state = 2
    state_dim = 4
    encoder = DiscreteOneHot(gym.spaces.Discrete(state_dim))
    one_hot_vector = encoder.encode(state)
    assert isinstance(one_hot_vector, torch.Tensor)
    assert one_hot_vector.shape == (state_dim,)
    assert one_hot_vector.sum().item() == 1.0

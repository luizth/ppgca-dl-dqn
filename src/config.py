import uuid
from dataclasses import dataclass


@dataclass
class JobConfig:
    ds: str
    eps: int
    lr: float
    exploration_decay: float
    use_conv: bool
    name: str = str(uuid.uuid4())[:8]
    arch: str = "DQN_Original"
    optim: str = "SGD"


def get():
    return [
        JobConfig(
            name="Acrobot-v1-MLP-" + str(uuid.uuid4())[:8],
            ds="Acrobot-v1",
            eps=10_000,
            lr=0.001,
            exploration_decay=0.99999,
            use_conv=False,
        ),
        JobConfig(
            name="CartPole-v1-MLP-" + str(uuid.uuid4())[:8],
            ds="CartPole-v1",
            eps=10_000,
            lr=0.001,
            exploration_decay=0.99999,
            use_conv=False,
        ),
        JobConfig(
            name="MountainCar-v0-MLP-" + str(uuid.uuid4())[:8],
            ds="MountainCar-v0",
            eps=10_000,
            lr=0.001,
            exploration_decay=0.99999,
            use_conv=False,
        ),
        JobConfig(
            name="LunarLander-v3-MLP-" + str(uuid.uuid4())[:8],
            ds="LunarLander-v3",
            eps=10_000,
            lr=0.001,
            exploration_decay=0.99999,
            use_conv=False,
        ),
        JobConfig(
            name="CliffWalking-v1-MLP-" + str(uuid.uuid4())[:8],
            ds="CliffWalking-v1",
            eps=10_000,
            lr=0.001,
            exploration_decay=0.99999,
            use_conv=False,
        ),
        JobConfig(
            name="FrozenLake-v1-4x4-MLP-" + str(uuid.uuid4())[:8],
            ds="FrozenLake-v1-4x4",
            eps=10_000,
            lr=0.001,
            exploration_decay=0.99999,
            use_conv=False,
        ),
        JobConfig(
            name="FrozenLake-v1-8x8-MLP-" + str(uuid.uuid4())[:8],
            ds="FrozenLake-v1-8x8",
            eps=10_000,
            lr=0.001,
            exploration_decay=0.99999,
            use_conv=False,
        ),
    ]

import uuid
from dataclasses import dataclass


@dataclass
class JobConfig:
    ds: str
    eps: int
    lr: float
    exploration_decay: float
    use_conv: bool
    name: str
    arch: str = "DQN_Original"
    optim: str = "SGD"

    def __post_init__(self):
        if not self.name:
            self.name = str(uuid.uuid4())[:8]


def get():
    """Jobs do experimento: os quatro ambientes de controle.

    O decay do epsilon e aplicado por passo (agent.py:203), mas as curvas do
    wandb sao por episodio — e o tamanho medio do episodio varia ~20x entre
    estes ambientes. Medido com politica aleatoria (30 episodios cada):
    CartPole 23 passos, LunarLander 95, MountainCar 200, Acrobot 500.

    Por isso o decay e escolhido por ambiente. O numero de passos ate o
    epsilon atingir o piso de 0.1 (min_exploration_rate, main.py:149) e

        passos = ln(0.1) / ln(exploration_decay)

    ┌────────────────┬─────────┬──────────────┬───────────────────────────────┐
    │ env            │  decay  │ ε=0.1 em     │ ~episodios explorando         │
    ├────────────────┼─────────┼──────────────┼───────────────────────────────┤
    │ CartPole-v1    │ 0.9999  │  23k passos  │ ~600 (crescem conforme aprende)│
    │ Acrobot-v1     │ 0.99995 │  46k passos  │ ~92                            │
    │ LunarLander-v3 │ 0.99998 │ 115k passos  │ ~1 200                         │
    │ MountainCar-v0 │ 0.99999 │ 230k passos  │ ~1 151                         │
    └────────────────┴─────────┴──────────────┴───────────────────────────────┘

    Com o 0.99999 anterior em todos, o CartPole precisaria de ~10 173
    episodios so para sair da fase de exploracao — mais do que o run inteiro.

    eps: dimensionado para sobrar treino depois da exploracao, sem os 10 000
    episodios anteriores (que davam 2M a 10M passos por job).

    lr: 0.001 e um bom valor para Adam. O otimizador esta fixo em SGD
    (agent.py:69) e o campo optim abaixo nao e lido pelo codigo.
    """
    return [
        JobConfig(
            name="CartPole-v1-MLP-" + str(uuid.uuid4())[:8],
            ds="CartPole-v1",
            eps=2_000,
            lr=0.001,
            exploration_decay=0.9999,
            use_conv=False,
        ),
        JobConfig(
            name="Acrobot-v1-MLP-" + str(uuid.uuid4())[:8],
            ds="Acrobot-v1",
            eps=1_000,
            lr=0.001,
            exploration_decay=0.99995,
            use_conv=False,
        ),
        JobConfig(
            name="LunarLander-v3-MLP-" + str(uuid.uuid4())[:8],
            ds="LunarLander-v3",
            eps=3_000,
            lr=0.001,
            exploration_decay=0.99998,
            use_conv=False,
        ),
        JobConfig(
            name="MountainCar-v0-MLP-" + str(uuid.uuid4())[:8],
            ds="MountainCar-v0",
            eps=3_000,
            lr=0.001,
            exploration_decay=0.99999,
            use_conv=False,
        ),
    ]


def tabular():
    """Ambientes tabulares — fora do experimento atual, preservados aqui.

    CliffWalking-v1 nao tem max_episode_steps registrado no gymnasium, entao o
    episodio so termina no objetivo. FrozenLake roda com is_slippery=False
    (main.py:64-77), ou seja, deterministico.
    """
    return [
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

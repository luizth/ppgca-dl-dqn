
# Deep Q Network experiments

Experiments using a Deep Q Network architecture on control environment from gymnasium benchmarks.

## Dependencies

We recommend installing the project with [uv](https://github.com/astral-sh/uv) by simply running
```bash
$ uv sync
```

This project uses `wandb` for monitoring training data. To run you must first login
```bash
$ wandb login
```

## Environment Setup

Before running experiments, you need to configure environment variables by creating a `.env` file:
```bash
$ cp .env.example .env
```

Then edit the `.env` file with your configuration:

- **WANDB_ENTITY**: Your Weights & Biases username or team name (sign up at [wandb.ai](https://wandb.ai) if you don't have an account)
- **WANDB_PROJECT**: The name of your W&B project for tracking experiments
- **RAY_NUM_CPUS**: Number of CPUs to allocate for Ray distributed computing (defaults to 20)

Example `.env` file:
```bash
WANDB_ENTITY=myusername
WANDB_PROJECT=dqn-experiments
RAY_NUM_CPUS=20
```

## Running experiments

The framework allows to configure a Deep Q Network with two different observations
- First, an agent uses the internal state of the environment. In that case, the observation is
passed directly from the environment to the agent's network.
- Secondly, an agent construct the features through image processing of frames extracted from the
environment. This is the framework described in the original paper that uses a CNN to build a
representation of Atari games.

Both approaches work. The first is faster to train and experiment with. Whereas the second is more
general because it do not rely on designed features of the environment.

The experiments are defined in `config.py` using the `Job` interface. Finally, to run the application
```bash
$ uv run ./src/main.py
```

## Reference

The Deep Q Network architecture was introduced in a paper from Nature [1].

[1] Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." nature 518.7540 (2015): 529-533.

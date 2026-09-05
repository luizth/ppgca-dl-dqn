import gymnasium as gym
import numpy as np
import torch
import wandb
import os
from dataclasses import asdict
from dotenv import load_dotenv

from network import DQN, CNN, MLP
from agent import DeepQLearning
from pre_processing import ImagePreprocessor

import config

# Load environment variables from .env file
load_dotenv()

# O Ray é ativado com USE_RAY=1. Com RAY_NUM_CPUS=1 não há paralelismo
# mas produz o overhead de driver + raylet + object store
USE_RAY = os.getenv("USE_RAY", "0") == "1"
OUT_DIR = os.getenv("OUT_DIR", "out")

os.makedirs(OUT_DIR, exist_ok=True)


def run_job(config: config.JobConfig):

    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity=os.getenv("WANDB_ENTITY"),
        # Set the wandb project where this run will be logged.
        project=os.getenv("WANDB_PROJECT"),
        # Set the name of the run, which is used to identify this run in the wandb app.
        name=config.name,
        # Track hyperparameters and run metadata.
        config={
            "architecture": config.arch,
            "dataset": config.ds,
            "episodes": config.eps,
            "optim": config.optim,
            "learning_rate": config.lr,
            "reward_clip": config.reward_clip,
            "action_repeat": config.action_repeat,
            "scale_exploration": config.scale_exploration,
        },
    )

    # Define metrics to track
    run.define_metric("episode")
    run.define_metric("loss", step_metric="global_step")
    run.define_metric("epsilon", step_metric="global_step")
    run.define_metric("steps_to_end_episode", step_metric="episode")
    run.define_metric("episode_reward", step_metric="episode")

    # Env
    if config.ds == "CartPole-v1":
        env = gym.make(
            "CartPole-v1",
            sutton_barto_reward=True,
            render_mode="rgb_array"
        )
    elif config.ds == "LunarLander-v3":
        env = gym.make(
            "LunarLander-v3",
            continuous=False,
            render_mode="rgb_array"
        )
    elif config.ds == "FrozenLake-v1-4x4":
        env = gym.make(
            "FrozenLake-v1",
            map_name="4x4",
            render_mode="rgb_array",
            is_slippery=False
        )
    elif config.ds == "FrozenLake-v1-8x8":
        env = gym.make(
            "FrozenLake-v1",
            map_name="8x8",
            render_mode="rgb_array",
            is_slippery=False
        )
    else:
        env = gym.make(
            config.ds,
            render_mode="rgb_array"
        )
    env.reset()

    # The input are the state features
    if isinstance(env.observation_space, gym.spaces.Discrete):
        in_size = env.observation_space.n
    elif isinstance(env.observation_space, gym.spaces.Box):
        in_size = env.observation_space.shape[0]
    else:
        raise NotImplementedError(f"Network input size not implemented for this type of state space ({type(env.observation_space)}).")

    # Default do not use preprocessor
    preprocessor = None

    if not config.use_conv:

        # Network - we use a MLP as Q-network
        # The output are the Q-values of each action
        net = MLP(
            in_size,
            env.action_space.n,
            hidden_layers=1,
            hidden_units=[128],
        )

    else:
        # Set frame preprocessor
        m = 4  # Number of frames to stack to conv net
        state_dim = 84
        preprocessor = ImagePreprocessor(m=m, frame_size=state_dim)

        # Network - we use a ConvNet + MLP as Q-network
        # CNN to process image input
        cnn = CNN(
            in_channels=m,
            conv_layers=3,
            channels=[32, 64, 64],
            kernel_sizes=[8, 4, 3],
            strides=[4, 2, 1],
        )

        with torch.no_grad():
            n_flat = cnn(torch.zeros(1, m, state_dim, state_dim)).flatten(1).shape[1]

        # MLP to process features from CNN
        mlp = MLP(
            in_features=n_flat,
            out_features=env.action_space.n,
            hidden_layers=1,
            hidden_units=[512],
        )

        # DQN Network
        net = DQN(cnn=cnn, mlp=mlp)

    # Agent
    agent = DeepQLearning(
        env=env,
        Q_network=net,
        eb_size=10000,
        batch_size=32,
        update_Q_target_every=100,
        number_of_states=in_size,
        number_of_actions=env.action_space.n,
        learning_rate=config.lr,
        discount_factor=0.99,
        exploration_rate=1.0,
        min_exploration_rate=0.1,
        exploration_decay=config.exploration_decay,
        preprocessor=preprocessor,
        reward_clip=config.reward_clip,
        action_repeat=config.action_repeat,
        scale_exploration=config.scale_exploration,
    )

    # Reset
    agent.reset()

    # Create tqdm progress bar outside the loop
    # pbar = tqdm.tqdm(total=config.eps, desc="Episodes")

    # Global step counter
    global_step = 0

    # Historico por episodio, para o resumo devolvido ao final
    episode_rewards = []
    episode_lengths = []

    # Train the agent
    for i in range(config.eps):

        done = False
        steps = 0
        episode_reward = 0
        while not done:
            # Perform a single training step
            losses, reward, steps, epsilon, done = agent.train_one_step()

            # Accumulate reward
            episode_reward += reward

            # Log intra-episode metrics to wandb
            # for loss in losses:
            #     run.log({"step": steps, "loss": loss})
            # run.log({"reward": reward}, step=global_step)
            run.log({
                "global_step": global_step,  # x-axis
                "epsilon": epsilon,
                "loss": np.average(losses)
            })

            # Increment global step counter
            global_step += 1

        # Log episode metrics to wandb
        run.log({
            "episode": i,  # x-axis
            "steps_to_end_episode": steps,
            "episode_reward": episode_reward
        })

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        # Update tqdm progress bar
        # pbar.update(1)
        # pbar.set_postfix({"reward": f"{episode_reward:.2f}"})

    # Salva os pesos da rede Q aprendida
    checkpoint = os.path.join(OUT_DIR, f"{config.name}.pt")
    torch.save(
        {
            "state_dict": agent.Q.state_dict(),
            "job": asdict(config),
            "in_size": in_size,
            "number_of_actions": int(env.action_space.n),
            "episodes_trained": config.eps,
        },
        checkpoint,
    )

    wandb_url = run.url

    # Finish the run and upload any remaining data.
    run.finish()
    env.close()

    last = slice(-100, None)  # ultimos 100 episodios
    return {
        "name": config.name,
        "ds": config.ds,
        "episodes": config.eps,
        "reward_last_100": float(np.mean(episode_rewards[last])),
        "steps_last_100": float(np.mean(episode_lengths[last])),
        "checkpoint": checkpoint,
        "wandb_url": wandb_url,
    }


if __name__ == "__main__":

    configs = config.get()

    if USE_RAY:
        import ray

        ray.init(
            num_cpus=int(os.getenv("RAY_NUM_CPUS", "20")),
            runtime_env={"working_dir": "."}  # Use current dir directly, no packaging
        )
        run_job_remote = ray.remote(num_cpus=1)(run_job)  # allocate 1 core for each job
        results = ray.get([run_job_remote.remote(cfg) for cfg in configs])
    else:
        results = [run_job(cfg) for cfg in configs]

    # run_job devolve um resumo por job; antes results.txt so recebia None
    with open(os.path.join(OUT_DIR, "results.txt"), "w") as f:
        f.write(f"{'name':38s} {'env':16s} {'eps':>6s} {'reward_100':>11s} {'steps_100':>10s}  checkpoint\n")
        for r in results:
            f.write(
                f"{r['name']:38s} {r['ds']:16s} {r['episodes']:6d} "
                f"{r['reward_last_100']:11.1f} {r['steps_last_100']:10.1f}  {r['checkpoint']}\n"
            )
        f.write("\n")
        for r in results:
            f.write(f"{r['name']}: {r['wandb_url']}\n")

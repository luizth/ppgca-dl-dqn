import gymnasium as gym
import numpy as np
import wandb

from network import DQN, CNN, MLP
from agent import DeepQLearning
from pre_processing import ImagePreprocessor

import ray
import config


ray.init(
    num_cpus=20,
    runtime_env={"working_dir": "."}  # Use current dir directly, no packaging
)

@ray.remote(num_cpus=1)  # allocate 1 core for each job
def run_job(config: config.JobConfig):

    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="luizthomasini-unisinos",
        # Set the wandb project where this run will be logged.
        project="DQN-experiments",
        # Set the name of the run, which is used to identify this run in the wandb app.
        name=config.name,
        # Track hyperparameters and run metadata.
        config={
            "architecture": config.arch,
            "dataset": config.ds,
            "episodes": config.eps,
            "optim": config.optim,
            "learning_rate": config.lr,
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
        # Network - we use a ConvNet + MLP as Q-network
        # CNN to process image input
        cnn = CNN(
            in_channels=m,
            conv_layers=3,
            channels=[32, 64, 64],
            kernel_sizes=[8, 4, 3],
            strides=[4, 2, 1],
        )

        # MLP to process features from CNN
        mlp = MLP(
            in_features=64 * 7 * 7,  # Assuming input image size after CNN layers
            out_features=env.action_space.n,
            hidden_layers=1,
            hidden_units=[512],
        )

        # Set frame preprocessor
        m = 4  # Number of frames to stack to conv net
        state_dim = 84
        preprocessor = ImagePreprocessor(m=m, frame_size=state_dim)

        # DQN Network
        net = DQN(
            cnn=cnn,
            mlp=mlp,
        )

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
        preprocessor=preprocessor
    )

    # Reset
    agent.reset()

    # Create tqdm progress bar outside the loop
    # pbar = tqdm.tqdm(total=config.eps, desc="Episodes")

    # Global step counter
    global_step = 0

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

        # Update tqdm progress bar
        # pbar.update(1)
        # pbar.set_postfix({"reward": f"{episode_reward:.2f}"})

    # Finish the run and upload any remaining data.
    run.finish()


configs = config.get()

futures = [run_job.remote(cfg) for cfg in configs]
results = ray.get(futures)

with open("results.txt", "w") as f:
    for result in results:
        f.write(f"{result}\n")

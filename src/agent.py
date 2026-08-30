import random
import tqdm
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from baseline import QLearning
from eb import Experience, ExperienceBuffer
from feature_construction import DiscreteOneHot, ContinuousNormalized
from pre_processing import ImagePreprocessor

State = Any
State_Dim = Any
State_Features = torch.Tensor


class DeepQLearning(QLearning):

    def __init__(
            self,
            env: gym.Env,
            Q_network: nn.Module,
            eb_size: int,
            batch_size: int,
            update_Q_target_every: int,
            # Q-Learning params
            number_of_states: int,  # Assumes discrete state space
            number_of_actions: int,  # Assumes discrete action space
            learning_rate: float = 0.01,  # Learning rate, for Q-network optimizer
            discount_factor: float = 0.99,  # MDP Discount factor
            exploration_rate: float = 1.0,  # Exploration rate - epsilon
            min_exploration_rate: float = 0.1,
            exploration_decay: float = 0.99,
            preprocessor: ImagePreprocessor = None,):

        super().__init__(
            number_of_states,
            number_of_actions,
            learning_rate,
            discount_factor,
            exploration_rate,
            min_exploration_rate,
            exploration_decay,)

        # Store env
        self._env = env

        # State Repr Encoder (ϕ phi)
        if isinstance(env.observation_space, gym.spaces.Discrete):
            self.encode: Callable[[State, State_Dim], State_Features] = DiscreteOneHot.encode
            self.state_dim = env.observation_space.n
        elif isinstance(env.observation_space, gym.spaces.Box):
            self.encode: Callable[[State, State_Dim], State_Features] = ContinuousNormalized.encode
            self.state_dim = env.observation_space.shape[0]
        else:
            raise NotImplementedError(f"State encoding not implemented for this type of state space ({type(env.observation_space)}).")

        self.preprocessor = preprocessor

        # Q-Value Model
        self.Q = Q_network
        self.Q_target = Q_network.copy()
        self._Q_initial = Q_network.copy()  # Keep a copy of the initial Q-network for resets

        # Optim
        # Use self.Q.parameters() instead of model.parameters()
        self.optimizer = optim.SGD(self.Q.parameters(), lr=learning_rate) # Used the learning_rate from init

        # Loss
        self.lossfn = nn.MSELoss()

        # Experience Buffer
        self.eb = ExperienceBuffer(eb_size)
        self.batch_size = batch_size

        # Q-target update frequency
        self.C = update_Q_target_every

        # Store steps taken and current state for single-step training
        self._number_of_steps_taken_in_episode = 0
        self._global_step = 0
        self._current_state, _ = self._env.reset()

    def reset(self):
        super().reset()
        # Reset Q-network to initial weights
        self.Q = self._Q_initial.copy()
        self.Q_target = self._Q_initial.copy()
        # Reset optimizer
        self.optimizer = optim.SGD(self.Q.parameters(), lr=self.learning_rate)
        # Reset current state
        self._current_state, _ = self._env.reset()
        # Reset counters
        self._number_of_steps_taken_in_episode = 0
        self._global_step = 0

    def _update_Q_target(self):
        self.Q_target = self.Q.copy()

    def _get_Q_values(self, state_features) -> torch.Tensor:
        return self.Q(state_features).squeeze()

    def _get_Q_target_values(self, state_features) -> torch.Tensor:
        return self.Q_target(state_features).squeeze()

    def choose_action(self, state_features) -> int:
        """Choose an action based on the exploration-exploitation trade-off"""
        if random.uniform(0, 1) < self.exploration_rate:
            # Exploration: choose a random action
            return np.random.choice(self.actions)
        else:
            # Exploitation: choose the best action based on Q-values
            q_values = self.Q(state_features)
            actions_i = np.random.choice( np.flatnonzero(q_values == q_values.max()) )
            return self.actions[actions_i]

    def choose_action_greedy(self, state_features) -> int:
        """Choose an action based on the Q-values"""
        q_values = self.Q(state_features)
        actions_i = np.random.choice( np.flatnonzero(q_values == q_values.max()) )
        return self.actions[actions_i]

    def update_Q_network(self):
        """Update the Q-value for the given state and option index"""

        # Store losses for logging
        losses = []

        # Store ys and yhats for loss calculation
        # ys = []
        # yhats = []

        minibatch = self.eb.sample(self.batch_size)

        for exp in minibatch:
            action = exp.action
            reward = exp.reward
            done = exp.done

            # Those are tensors
            state_features = exp.state
            next_state_features = exp.next_state

            # Calculate TD target - bootstrap
            with torch.no_grad(): # No need to track gradients for target calculation
                if done:
                    td_target = torch.tensor(reward, dtype=torch.float32)
                else:
                    next_q_values = self._get_Q_target_values(next_state_features)
                    td_target = torch.tensor(reward, dtype=torch.float32) + self.discount_factor * torch.max(next_q_values)

            # Get the current Q-value prediction
            q_values = self._get_Q_values(state_features)
            q_value = q_values[action]

            # Compute loss
            loss = self.lossfn(td_target, q_value)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Store loss
            losses.append(loss.item())

        return losses

    def train_one_step(self):
        """Perform one step of training"""

        # Build state representation
        if self.preprocessor is not None:
            self._current_state = self._env.render()
            state_features = self.preprocessor.get_state_tensor(self._current_state)
        else:
            state_features = self.encode(self._current_state, self.state_dim)

        # Choose action based on current state
        action = self.choose_action(state_features)

        # Step
        next_state, reward, done, trunc, info = self._env.step(action)

        # Clip negative reward at -1, positive reward at 1
        reward = -1.0 if reward < 0 else (1.0 if reward > 0 else 0.0)

        # Build next state representation
        if self.preprocessor is not None:
            next_state = self._env.render()
            next_state_features = self.preprocessor.get_state_tensor(next_state)
        else:
            next_state_features = self.encode(next_state, self.state_dim)

        # Store experience
        self.eb.add(Experience(state_features, action, reward, next_state_features, done))

        # Perform Q updates
        losses = self.update_Q_network()

        # Increment step counter
        self._number_of_steps_taken_in_episode += 1
        self._global_step += 1

        # Update epsilon
        self.decay_exploration_rate()

        # Every C steps we update target Q
        if self._global_step % self.C == 0:
            self._update_Q_target()

        # Update current state
        if done or trunc:
            _episode_steps = self._number_of_steps_taken_in_episode
            self._current_state, _ = self._env.reset()
            self._number_of_steps_taken_in_episode = 0
            return losses, reward, _episode_steps, self.exploration_rate, True  # done
        else:
            self._current_state = next_state

        return losses, reward, self._number_of_steps_taken_in_episode, self.exploration_rate, False  # done

    def train(
            self,
            number_of_episodes: int,
            max_number_of_steps: int):
        self.reset()

        for _ in tqdm(range(number_of_episodes)):
            done = False
            i = 0

            state, info = self._env.reset()
            while i <= max_number_of_steps and not done:

                # Build state representation
                state_features = self.encode(state, self.state_dim)

                # Choose action based on current state
                action = self.choose_action(state_features)

                # Step
                next_state, reward, done, trunc, info = self._env.step(action)

                # Build next state representation
                next_state_features = self.encode(next_state, self.state_dim)

                # Store experience
                self.eb.add(Experience(state_features, action, reward, next_state_features, done))

                # Perform Q updates
                self.update_Q_network()

                # Every C steps we update target Q
                if i % self.C == 0:
                    self._update_Q_target()

                # Step
                state = next_state
                i += 1

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
            self.encoder = DiscreteOneHot(env.observation_space)
        elif isinstance(env.observation_space, gym.spaces.Box):
            self.encoder = ContinuousNormalized(env.observation_space)
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
        # Reset replay buffer
        self.eb = ExperienceBuffer(self.eb.max_lenght)
        # Reset preprocessor state buffer if it exists
        if self.preprocessor is not None:
            self.preprocessor.reset()
        # Reset current state
        self._current_state, _ = self._env.reset()
        # Reset counters
        self._number_of_steps_taken_in_episode = 0
        self._global_step = 0

    def _update_Q_target(self):
        self.Q_target = self.Q.copy()

    def choose_action(self, state_features) -> int:
        """Choose an action based on the exploration-exploitation trade-off"""
        if random.uniform(0, 1) < self.exploration_rate:
            # Exploration: choose a random action
            return np.random.choice(self.actions)
        else:
            # Exploitation: choose the best action based on Q-values
            q_values = self.Q(state_features.unsqueeze(0))[0]  # add batch dimension for the network
            actions_i = np.random.choice( np.flatnonzero(q_values == q_values.max()) )
            return self.actions[actions_i]

    def choose_action_greedy(self, state_features) -> int:
        """Choose an action based on the Q-values"""
        q_values = self.Q(state_features.unsqueeze(0))[0]  # add batch dimension for the network
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

        # Empilhar o minibatch em tensores para processamento em lote
        # torch.stack will create a tensor of shape (batch_size [32], 1, 4, 84, 84)
        states      = torch.stack([e.state for e in minibatch])
        next_states = torch.stack([e.next_state for e in minibatch])
        actions     = torch.tensor([e.action for e in minibatch])
        rewards     = torch.tensor([e.reward for e in minibatch], dtype=torch.float32)
        dones       = torch.tensor([e.done for e in minibatch], dtype=torch.float32)

        # DQN - um passo de gradiente sobre a média do minibatch
        with torch.no_grad():
            targets = rewards + self.discount_factor * self.Q_target(next_states).max(1).values * (1 - dones)

        q = self.Q(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.lossfn(q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return [loss.item()]

    def train_one_step(self):
        """Perform one step of training"""

        # Try to get the last state from the experience buffer, if available
        try:
            if self._number_of_steps_taken_in_episode == 0:
                raise IndexError("Starting a new episode, previous state in experience buffer is from past episode.")
            state_features = self.eb.peak().next_state
        except (IndexError, ValueError):
            # If the buffer is empty, build state representation from env observation
            if self.preprocessor is not None:
                self._current_state = self._env.render()
                state_features = self.preprocessor.get_state_tensor(self._current_state)
            else:
                state_features = self.encoder.encode(self._current_state)

        # Choose action based on current state
        with torch.no_grad():
            action = self.choose_action(state_features)

        # Step
        next_state, reward, done, trunc, info = self._env.step(action)

        # Clip reward for learning
        if self._env.spec.id in ["LunarLander-v2", "LunarLander-v3"]:
            reward_clip = np.clip(reward, -1.0, 1.0)  # distinguish landing from flying
        else:
            # Reward signal (Mnih et al. 2015)
            reward_clip = -1.0 if reward < 0 else (1.0 if reward > 0 else 0.0)

        # Build next state representation
        if self.preprocessor is not None:
            next_state = self._env.render()
            next_state_features = self.preprocessor.get_state_tensor(next_state)
        else:
            next_state_features = self.encoder.encode(next_state)

        # Store experience
        self.eb.add(Experience(state_features, action, reward_clip, next_state_features, done))

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
            if self.preprocessor is not None:
                self.preprocessor.reset()  # reset the preprocessor state buffer
            self._current_state, _ = self._env.reset()
            self._number_of_steps_taken_in_episode = 0
            return losses, reward, _episode_steps, self.exploration_rate, True  # done
        else:
            self._current_state = next_state

        return losses, reward, self._number_of_steps_taken_in_episode, self.exploration_rate, False  # done

""" old: def update_Q_network(self):

    Análise:

    dW = o quanto os pesos andaram.

    Pense na rede como um ponto num mapa. Treinar é dar passos com esse ponto.
    dW é a seta de onde ele estava até onde ele foi parar depois de um passo do ambiente.
    ||dW|| é só o comprimento dessa seta: a distância percorrida.

    O cosseno = se as duas setas apontam para o mesmo lado.
    - 1.0 → mesmíssima direção
    - 0.0 → uma para o norte, outra para o leste
    - -1.0 → direções opostas

    Deu 0.83: apontam mais ou menos para o mesmo lado, mas torto.
    Então o laço não erra só na distância - ele vai para um lugar um pouco diferente.

    O 3,4x.
    32 amigos te dão conselho sobre para onde andar.
    - Jeito certo (minibatch): você ouve os 32, tira a média, dá um passo.
    - Jeito do código: você obedece o amigo 1 por inteiro, anda. Aí ouve o amigo 2,
    do lugar novo, e anda. E assim por diante, 32 vezes.

    No fim você parou 3,4x mais longe, e num ponto meio torto.

    E por que não dá para dizer "é só um learning rate 32x maior"?
    Porque cada passo muda o chão do próximo.
    O amigo 5 dá um conselho diferente do que daria se você não tivesse andado antes.
    Então o total não é 32 x nada, depende de quais amigos calharam de estar no grupo
    e de onde a rede está naquele momento. Hoje deu 3,4x, amanhã dá outro número.

    É esse o problema: o tamanho do passo vira uma variável escondida
    que ninguém controla nem consegue prever.

    # 32 atualizações online sequenciais -> errado
    # além disso, as amostras NÃO vem da mesma rede, porque a cada passo a rede é atualizada,
    # então o Q_target muda a cada passo, e o Q muda a cada passo.
    #   sequencial: || dW || = 0.33215
    #   minibatch real: || dW || = 0.09825
    #   cosseno entre as direcoes: 0.8287 -> 1.0 = mesma direção
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
"""

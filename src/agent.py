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
            preprocessor: ImagePreprocessor = None,
            reward_clip: bool = True,
            action_repeat: int = 1,  # k=1 one action per step, k>1 hold action for k steps
            scale_exploration: bool = True,  # epsilon como fracao do tempo explorando
            ):

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

        # Clipping de recompensa (Mnih et al. 2015)
        # e.g. no LunarLander não usamos o clip para preservar o +-100 do pouso.
        self.reward_clip = reward_clip

        # Repeticao de ação (k) / frame-skipping (Mnih et al. 2015; Bellemare et al. 2012 (k=4))
        # A exploracao epsilon-greedy sorteia uma acao nova a cada passo.
        # Em num ambiente como o MountainCar os sorteios se cancelam:
        # o carro so treme no fundo do vale. Segurar a acao exploratoria
        # por k passos torna a exploracao temporalmente correlacionada.

        # e.g. politica aleatória, limite de 200 passos, chegadas ao topo:
        # k=1 -> 0/500, k=20 -> 58/500, k=30 -> 97/500.

        # O mecanismo tem vantagem: no custo computacional (menos queries a rede Q);
        # e consistência na exploração (menos ruído em ações aleatórias)
        self.action_repeat = action_repeat
        self.scale_exploration = scale_exploration
        self._held_action = None
        self._hold_left = 0

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
        # Reset da acao segurada
        self._held_action = None
        self._hold_left = 0

    def _update_Q_target(self):
        self.Q_target = self.Q.copy()

    def _exploration_trigger_rate(self) -> float:
        """
        Taxa de gatilho para exploração

            p = ε / (k * (1 - ε) + ε)

        O numerador é o que queremos (ε). O denominador é "divide por k",
        o "k * (1 - ε)" é a divisão, e o "+ ε" é o ajuste fino que conta
        os passos gulosos que acontecem entre os blocos, que também
        consomem tempo.

        ε: epsilon, taxa de exploração (exploration_rate)
        k: action_repeat, passos que a acao exploratoria e mantida

        Se ε=0.1 na definição, 10% dos passos deveriam ser exploratórios.
        Quando k>1, criamos blocos de k passos exploratórios, que melhora
        a exploração no caso Mountain Car, porém aumenta o tempo explorando.
        Com ε=0.1 e k=30: 77% dos passos saem exploratorios, nao 10%.

        Sorteando com probabilidade p, cada ciclo tem p*k passos exploratorios
        em p*k + (1 - p) passos totais, entao a fracao do tempo explorando e

            f = k*p / (1 - p + k*p)

        Igualando f a epsilon e isolando p:

            p = eps / (k*(1 - eps) + eps)

        Com k=1 (bloco de 1 passo, sem repetição) -> p = ε. O ε-greedy normal.
        Com ε=1 (explorar o tempo todo) -> p = 1. Sempre inicia bloco.

        Com scale_exploration=False fica o modo ingênuo, p = eps: o epsilon
        passa a significar "probabilidade de iniciar um bloco", e a fracao do
        tempo explorando sobe junto com k.

        No MountainCar, taxa de sucesso no modo ingênuo contra a fracao do tempo explorando:
        - 1500 episodios, k=4: 0.3% contra 19.2%.
        - 3000 episodios, k=30: 32.2% contra 88.1%.

        Em resumo: ε diz quanto tempo explorar; p diz com que frequência iniciar,
        dado que cada bloco dura k passos. Com k>1 -> p > ε, e a fracao do tempo
        explorando é maior que ε.
        """
        eps = self.exploration_rate
        k = self.action_repeat
        if not self.scale_exploration:
            return eps
        return eps / (k * (1.0 - eps) + eps)

    def choose_action(self, state_features) -> int:
        """Choose an action based on the exploration-exploitation trade-off"""
        # Action repeat: if we are still holding the previous action, return it
        if self._hold_left > 0:
            self._hold_left -= 1
            return self._held_action
        if random.uniform(0, 1) < self._exploration_trigger_rate():
            # Exploration: choose a random action, e segura por action_repeat passos
            self._held_action = np.random.choice(self.actions)
            self._hold_left = self.action_repeat - 1
            return self._held_action
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
        if not self.reward_clip:
            reward_clip = reward
        elif self._env.spec.id in ["LunarLander-v2", "LunarLander-v3"]:
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
            self._held_action = None
            self._hold_left = 0  # nao carrega a acao segurada entre episodios
            return losses, reward, _episode_steps, self.exploration_rate, True  # done
        else:
            self._current_state = next_state

        return losses, reward, self._number_of_steps_taken_in_episode, self.exploration_rate, False  # done

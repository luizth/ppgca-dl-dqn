import random
import numpy as np
from tqdm import tqdm


# lets implement a tabular q learning and a nn q learning and compare them
class QLearning:

    def __init__(
            self,
            number_of_states: int,  # discrete
            number_of_actions: int = 4,
            learning_rate: float = 0.1,  # Learning rate
            discount_factor: float = 0.99,  # Discount factor
            exploration_rate: float = 1.0,  # Exploration rate
            min_exploration_rate: float = 0.1,
            exploration_decay: float = 0.99,):

        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_exploration_rate = exploration_rate
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay

        self.actions = list(range(number_of_actions))
        # in DQN -> Q is replaced by a neural network (Q initializes and gets overwritten by the neural network)
        self.Q = np.zeros( (number_of_states, number_of_actions) )

    def reset(self):
        """Reset the Q-table and exploration rate"""
        self.Q = np.zeros( (self.number_of_states, self.number_of_actions) )
        self.exploration_rate = self.initial_exploration_rate

    def decay_exploration_rate(self):
        """Decay the exploration rate to gradually shift from exploration to exploitation"""
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def choose_action(self, state) -> int:
        """Choose an action based on the exploration-exploitation trade-off"""
        if random.uniform(0, 1) < self.exploration_rate:
            # Exploration: choose a random action
            return np.random.choice(self.actions)
        else:
            # Exploitation: choose the best action based on Q-values
            actions_i = np.random.choice(np.flatnonzero(self.Q[state] == self.Q[state].max()))
            return self.actions[actions_i]

    def choose_action_greedy(self, state) -> int:
        """Choose an action based on the Q-values"""
        actions_i = np.random.choice(np.flatnonzero(self.Q[state] == self.Q[state].max()))
        return self.actions[actions_i]

    def update_q_value(self, state, action, reward, next_state, done, option_k=1):
        """Update the Q-value for the given state and option index"""
        if done:
            td_target = reward
        else:
            best_next_action_index = np.argmax(self.Q[next_state])
            td_target = reward + self.discount_factor * self.Q[next_state][best_next_action_index]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.learning_rate * td_error

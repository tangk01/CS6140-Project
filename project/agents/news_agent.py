from abstract.abstract_agent import AbstractAgent
import gymnasium as gym
import numpy as np
from collections import defaultdict


class NewsAgent(AbstractAgent):
    """
    Class NewsAgent

    Takes on both the real-information, and fake information based on type passed.
    """

    def __init__(
        self,
        env=None,
        learning_rate=0.1,
        discount=0.9,
        epsilon=0.1,
        epsilonDecay=0.0001,
    ):

        super().__init__(type)
        self.type = type

        if env is None:
            raise ValueError("Must provide gym.Env")

        self.env = env

        self.type = type
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.shape))

        self.learning = learning_rate  # Learning rate
        self.discount = discount  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilonDecay = epsilonDecay

    def select_action(self, state):
        """Select an action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            print(self.q_values)
            return int(np.argmax(self.q_values[state]))

    def update_q_value(self, state, action, reward, next_state):
        """Update the Q-value using the Q-learning update rule."""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning * td_error

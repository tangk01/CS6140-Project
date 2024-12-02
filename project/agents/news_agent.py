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
        agentType,
        trustLevel,
        state_space_size,
        env: gym.Env = None,
        learning_rate=0.1,
        discount=0.9,
        epsilon=0.1,
        epsilonDecay=0.0001,
    ):

        if env is None:
            raise ValueError("Must provide gym.Env")
        
        self.env = env
        self.agentType = agentType

        if self.agentType == "real-information" or self.agentType == "fake-information":
            self.trustLevel = 0
        self.trustLevel = trustLevel

        self.learning = learning_rate  
        self.discount = discount  
        self.epsilon = epsilon  
        self.epsilonDecay = epsilonDecay

        self.trustLevel = np.random.uniform(0, 1)
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.shape))
        self.q_table = np.zeros((state_space_size, 2)) # 2 actions: (1) send,  (0) dont send

        self.reward = 0
        self.penalty = 0

    def select_action(self, state):
        '''
        Actions = ["spread-info", "dont spread-info"]

        using epsilon-greedy strategy where ϵ = probability of choosing a random action &&
        1 - ϵ chooses action with highest q-value.
        '''
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[state]))

    def update_q_value(self, state, action, reward, next_state):
        """Update the Q-value using the Q-learning update rule."""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning * td_error


    def send_information(self):
        '''
        pass for abstraction
        '''
        pass

    def fact_check(self):
        '''
        newsAgent does not fact check. Pass for abstract class
        '''
        pass

    def get_type(self):
        return str(self.agentType)
    
    def __str__(self):
        return f"Agent type: {self.get_type()} penalty: {self.penalty} reward: {self.reward} trust level: {self.trustLevel}"

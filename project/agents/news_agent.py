
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
        env: gym.Env = None,
        learning_rate=0.1,
        discount=0.9,
        epsilon=0.3,
        epsilonDecay=0.01,
    ):

        if env is None:
            raise ValueError("Must provide gym.Env")
        
        self.env = env
        self.agentType = agentType

        self.learning = learning_rate  
        self.discount = discount  
        self.epsilon = epsilon  
        self.epsilonDecay = epsilonDecay

        self.trustLevel = trustLevel or np.random.uniform(0, 1)
        self.q_table = np.zeros((self.env.original_num_consumers, 1)) # 2 actions: (1) send,  (0) dont send

        self.influenced_consumers = []

    def select_action(self):
        '''
        Actions = ["spread-info", "dont spread-info"]

        using epsilon-greedy strategy where ϵ = probability of choosing a random action &&
        1 - ϵ chooses action with highest q-value.
        '''
        self.epsilon -= self.epsilonDecay
        print("QTABLE", self.q_table)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            action = []
            for val in self.q_table:
                if val >= 0:
                    action.append(1)
                else:
                    action.append(0)
            return action
        

    def update_q_value(self, state, action, qVal):
        """Update the Q-value using the Q-learning update rule."""
        if self.agentType == "fake-information":
            agent = -0.2
        else:
            agent = 0.2
        
        fake = state[1]
        real = state[0]
        
        # Loop through state and action to update
        for node, send in zip(range(len(action)), action):
            if send == 1:
                if node in fake:
                    self.q_table[node] -= agent * (qVal + 2)
                else:
                    self.q_table[node] -= agent * (qVal + 1)
                if node in real:
                    self.q_table[node] += agent * (qVal + 2)
                else:
                    self.q_table[node] += agent * (qVal + 1)
            else:
                self.q_table[node] += 0.2 * (qVal + 1)

    def factChecked(self, nodes):
        for node in nodes:
            self.q_table[node] -= 0.2 * 2



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
    
    def __repr__(self):
        return f"Agent type: {self.get_type()} QTable: {self.q_table} Trust Level: {self.trustLevel}"
    
from abc import ABC, abstractmethod
import gymnasium as gym
from gym import spaces
import numpy as np

class AbstractAgent(ABC):
    '''
    Abstract base class for agents

    Returns an agent of type super
    '''
    def __init__(self, agentType, trustLevel, env=None, learning_rate = 0.1, discount = 0.9, epsilon=0.1, epsilonDecay=0.0001):
        self.env = env

        self.agentType = agentType
        self.trustLevel = trustLevel
        self.agentInformationRecieved = []

        self.reward = 0
        self.pentalty = 0
        
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        
    @abstractmethod
    def select_action(self, state):   
        '''
        Select an action based on current state

        Args:
            State - the current state of the env.
        ''' 
        pass

    @abstractmethod
    def update_q_value(self, state, action, reward, next_state):
        '''
        Update the q-value based on some transition

        Args:
            state - current env state
            action - action taken
            reward - the amt reward recieved
            next_state - the state transitioned to
        '''
        pass

    @abstractmethod
    def send_information(self, dst):
        '''
        Agent decides to send info or to not send information to another agent
        '''
        pass

    @abstractmethod
    def fact_check(self, node):
        '''
        Fact check will only be applicable to the fact checking agent.

        Fact checking agents will check a consumer nodes information history and perform an audit against it 
        to assign reward/penalties to other real-info/fake-info agents.
        '''
        pass

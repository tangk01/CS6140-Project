from abc import ABC, abstractmethod
import gymnasium as gym

class AbstractAgent(ABC):
    '''
    Abstract base class for agents

    Returns an agent of type super
    '''
    def __init__(self, type):
        self.type = type
        
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
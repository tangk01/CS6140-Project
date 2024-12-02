from abstract.abstract_agent import AbstractAgent
import numpy as np

class FactCheckerAgent(AbstractAgent):
    def __init__(
            self, 
            agentType, 
            trustLevel, 
            env=None, 
            learning_rate=0.1, 
            discount=0.9, 
            epsilon=0.1, 
            epsilonDecay=0.0001
            ):
        
        super().__init__(agentType, trustLevel, env, learning_rate, discount, epsilon, epsilonDecay)

    
    def select_action(self, state, nodes):   
        pass
        

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

    def send_information(self, dst):
        '''
        Fact checkers do not send information.
        '''
        pass

    def fact_check(self, node):
        '''
        Fact check will only be applicable to the fact checking agent.

        Fact checking agents will check a consumer nodes information history and perform an audit against it 
        to assign reward/penalties to other real-info/fake-info agents.
        '''
        pass    

    def get_type(self):
        return str(self.agentType)





    


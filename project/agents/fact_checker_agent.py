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

    '''
    TODO: 

    fact checker should check a random consumer node that is storing the most information in n/w.
    '''
    def select_action(self, state, nodes):   
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[state]))
    
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

    def fact_check(self, fact_checker_agent, top_k=1):
        '''
        Fact check will only be applicable to the fact checking agent.

        Fact checking agents will check a consumer nodes information history and perform an audit against it 
        to assign reward/penalties to other real-info/fake-info agents.
        '''
        sorted_nodes = sorted(
        self.graph.nodes(data=True), key=lambda x: len(x[1]["storedInfo"]), reverse=True
    )
        for node, data in sorted_nodes[:top_k]:
            for info in data["storedInfo"]:
                src_agent, trust_in_src = info
                
                # Penalize fake agent if fake news was wrongly accepted
                if self.graph.nodes[src_agent]["agentType"] == "fake-information":
                    fact_checker_agent.reward += 1
                    self.graph.nodes[src_agent]["penalty"] += 1
                
                # Penalize fact-checker for mislabeling real news
                else:  
                    fact_checker_agent.penalty += 1 

    def get_type(self):
        return str(self.agentType)





    


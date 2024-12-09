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
        
        if env is None:
            raise ValueError("Must provide gym.Env")
        
        self.env = env
        self.agentType = agentType

    '''
    TODO: 

    fact checker should check a random consumer node that is storing the most information in n/w.
    '''
    def select_action(self, threshold=0.7):   
        max_stored_info_node = None
        max_stored_info_length = 0

        # Find the consumer node with the most interactions
        for node, data in self.env.graph.nodes(data=True):
            if data["agentType"] == "consumer":
                stored_info_length = len(data['interactions'])
                if stored_info_length > max_stored_info_length:
                    max_stored_info_length = stored_info_length
                    max_stored_info_node = data

        if max_stored_info_node is None or max_stored_info_length == 0:
            print("No consumer node to fact-check.")
            print(self.env.graph.nodes(data=True))
            return {}

        # Find the source agent with the highest influence
        influence_count = {}
        for src_agent in max_stored_info_node["interactions"]:
            print(src_agent)
            influence_count[src_agent] = influence_count.get(src_agent, 0) + src_agent.trustLevel

        actions = {k: 1 if v >= threshold else 0 for k, v in influence_count.items()}

        return actions or {}

    
    def update_q_value(self, state, action, reward, next_state):
        '''
        Update the q-value based on some transition

        Args:
            state - current env state
            action - action taken
            reward - the amt reward recieved
            next_state - the state transitioned to
        '''
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning * td_error

    def send_information(self, dst):
        '''
        Fact checkers do not send information.
        '''
        pass

    def fact_check(self, news_agent, top_k=1):
        src_data = self.env.get_node_from_agent(news_agent)
        
        if src_data["agentType"] == "fake-information":
            print(f"Fact-checker penalized news agent {src_data}.")
            src_data["penalty"] += 1
            news_agent.penalty += 1
            self.reward += 1  
            self.env.get_node_from_agent(self)["reward"] = self.reward

            return True                    

        else:
            print(f"Fact-checker found no fake news in node {news_agent}.")
            self.penalty += 1
            self.env.get_node_from_agent(self)["penalty"] = self.penalty

        return False 
    
    def get_type(self):
        return str(self.agentType)

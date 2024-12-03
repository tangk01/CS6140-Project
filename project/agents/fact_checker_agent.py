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

        # Find the node with the most interactions
        for node, data in self.env.graph.nodes(data=True):
            if data["agentType"] == "consumer":
                stored_info_length = len(data["storedInfo"])
                if stored_info_length > max_stored_info_length:
                    max_stored_info_length = stored_info_length
                    max_stored_info_node = node

        if max_stored_info_node is None or max_stored_info_length == 0:
            return 0  

        selected_consumer_data = self.env.graph.nodes[max_stored_info_node]
        influence_count = {}
        for src_agent, influence in selected_consumer_data["storedInfo"]:
            influence_count[src_agent] = influence_count.get(src_agent, 0) + float(influence)

        if not influence_count:
            return 0  

        max_influential_src = max(influence_count, key=influence_count.get)
        total_nodes = self.env.original_num_consumers
        network_reach = influence_count[max_influential_src] / total_nodes

        return 1 if network_reach > threshold else 0

    
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

    def fact_check(self, fact_checker_agent, top_k=1):
        max_stored_info_node = None
        max_stored_info_length = 0

        # Find the consumer node with the most interactions
        for node, data in self.env.graph.nodes(data=True):
            if data["agentType"] == "consumer":
                stored_info_length = len(data["storedInfo"])
                if stored_info_length > max_stored_info_length:
                    max_stored_info_length = stored_info_length
                    max_stored_info_node = node

        if max_stored_info_node is None or max_stored_info_length == 0:
            print("No consumer node to fact-check.")
            return

        # Find the source agent with the highest influence
        selected_consumer_data = self.env.graph.nodes[max_stored_info_node]
        influence_count = {}
        for src_agent, influence in selected_consumer_data["storedInfo"]:
            influence_count[src_agent] = influence_count.get(src_agent, 0) + float(influence)

        if not influence_count:
            print("No influence data to process.")
            return

        max_influential_src = max(influence_count, key=influence_count.get)

        # Check the agent type of the source
        src_data = self.env.graph.nodes[max_influential_src]
        if src_data["agentType"] == "fake-information":
            print(f"Fact-checker penalized fake news agent {max_influential_src}.")
            src_data["penalty"] += 1
            fact_checker_agent.reward += 1  
            self.env.graph.nodes[self.env.agent_to_node_map[fact_checker_agent]]["reward"] = fact_checker_agent.reward
        else:
            print(f"Fact-checker found no fake news in node {max_stored_info_node}.")
            fact_checker_agent.penalty += 1
            self.env.graph.nodes[self.env.agent_to_node_map[fact_checker_agent]]["penalty"] = fact_checker_agent.penalty

    def get_type(self):
        return str(self.agentType)

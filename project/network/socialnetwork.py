import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import networkx as nx
from agents.news_agent import NewsAgent
from agents.fact_checker_agent import FactCheckerAgent
import math


class SocialNetworkEnv(gym.Env):
    def __init__(self, numConsumer=10):
        super().__init__()
        self.graph = nx.DiGraph()
        self.original_num_consumers = numConsumer 
        self.numConsumers = numConsumer
        

        self.build_consumer_network(numConsumer)
        self.build_action_space()
        self.build_observation_space()
        

        self.network_size = self.numConsumers

        self.agent_to_node_map = {}
        self.edge_colors = {}
        self.node_edge_colors = {}


    def print_graph(self):
        for node in self.graph.nodes:
            neighbors = list(self.graph.neighbors(node))
            neighbor_str = ", ".join(map(str, neighbors)) if neighbors else "None"
            print(f"Node {node}: Points to -> {neighbor_str}")


    def __str__(self):
        return f"graph: {self.graph} num consumers: {self.numConsumers} total nw size: {self.network_size} action space: {self.action_space}"

    def build_consumer_network(self, numConsumer):
        '''
        Builds the network -- adds nodes of type="consumer". Inits each Node with type, trustLevel, information to store, reward and penalty.

        Parameters -- numConsumers: num. of consumers in network.

        Returns None
        '''
        for i in range(numConsumer):
            self.graph.add_node(i, agentType="consumer", trustLevel=0.0, storedInfo=[], reward=0, penalty=0, interactions=[])

        for _ in range(numConsumer * 2):
            src = np.random.randint(0, numConsumer)
            dst = np.random.randint(0, numConsumer)
            if src != dst:
                self.graph.add_edge(src, dst)

    def add_news_agents_to_network(self, agentType: NewsAgent):
        '''
        Adds a fake information agent to the network and connects it to all other consumer nodes.

        Params:
            numConsumers -- number of consumers it will connect edges to.
        '''
        new_node_id = self.numConsumers
        self.numConsumers += 1
        
        self.graph.add_node(new_node_id, agentType=agentType.get_type(),  qVal=0.0, trustLevel=0.0, reward=0, penalty=0)
        self.agent_to_node_map[agentType] = new_node_id  
        self.network_size += 1
        
        for node in self.graph.nodes:
                if node != new_node_id:
                    current_type = self.graph.nodes[node].get("agentType", "consumer")
                    if agentType.get_type() == "fake-information" and current_type != "real-information":
                        self.graph.add_edge(new_node_id, node)
                    elif agentType.get_type() == "real-information" and current_type != "fake-information":
                        self.graph.add_edge(new_node_id, node)


    # currently tied to one consumer randomly. Not based on anything atm.
    def add_fact_checker_to_network(self, agentType: FactCheckerAgent):
        '''
        Adds a fact checker to the network. fact checker will only be tied to a consumer.

    
        '''
        new_node_id = self.numConsumers
        self.numConsumers += 1

        self.graph.add_node(new_node_id, agentType = agentType.get_type(), qVal=0.0, trustLevel=0.0, reward=0, penalty=0)
        self.agent_to_node_map[agentType] = new_node_id
        self.network_size += 1
        
        for node in self.graph.nodes:
            if node != new_node_id:
                current_type = self.graph.nodes[node].get("agentType", "consumer")
                if agentType.get_type() == 'fact-checker' and current_type == "consumer":
                    self.graph.add_edge(new_node_id, node)
                    break

    def build_action_space(self):
        '''
        Creates an action space for each agent to be used within training

        (1): send info
        (0): dont send info

        builds an action space of shape = self.numConsumers
        randomly assigns 0/1 to each
        '''
        self.action_space = spaces.MultiBinary(self.numConsumers)
        return self.action_space

    def draw_sample_from_action_space(self):
        '''
        creates the sample space the agent will act upon. 

        i.e; if numConsumer = 4 this will return a vector [0, 1, 1, 0] for example
        where 0 != send, and 1 = send info
        '''
        action = self.action_space.sample()
        return action

    def build_observation_space(self):
        self.observation_space = spaces.Dict(
            {
                "trustLevels": spaces.MultiBinary(
                    self.numConsumers
                ),
            }
        )
        return self.observation_space

    # Reset Env
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset agents
        for node in self.graph.nodes:
            nodeType = self.graph.nodes[node]["agentType"]
            if nodeType == "consumer":
                self.graph.nodes[node]["trustLevel"] = 0.0
                self.graph.nodes[node]["storedInfo"] = []
                self.graph.nodes[node]["reward"] = 0
                self.graph.nodes[node]["penalty"] = 0
            elif nodeType == "fake-information":
                self.graph.nodes[node]["qVal"] = 0.0
                self.graph.nodes[node]["reward"] = 0
                self.graph.nodes[node]["penalty"] = 0

        # Generate random trust levels for all agents
        trustLevels = np.array(
            [self.graph.nodes[i]["trustLevel"] for i in range(self.numConsumers)]
        )

        # Return initial observation
        return {"trustLevels": trustLevels}, {}

    # Step Function for Env
    def step(self, action, agent : NewsAgent | FactCheckerAgent):
        '''
        Training loop for an agent

        Params:
            action: a set of actions the agent will take during training
            agent: an agent object to train

        Returns: {"trustLevels": trustLevels}, agent.reward, info
        '''
        
        agent_node = self.agent_to_node_map[agent]
        actionNode = self.graph.nodes[agent_node]

        visited = set()
        influenced = set()
        queue = []
        # print('orig', queue)

        # nodes_to_visit = []

        total_nodes = self.original_num_consumers
        for neighbor, sendInfo in zip(self.graph.neighbors(agent_node), action):
            if sendInfo == 1:
                queue.append((agent_node, neighbor))

        while queue:
            info = queue.pop(0)
            source = info[0]
            currentValue = info[1]
            # print('popping node:', currentValue, 'Queue before adding new neighbors:', queue)

            if currentValue in visited:
                continue
            visited.add(currentValue)
            currentNode = self.graph.nodes[currentValue]
            
            if actionNode["agentType"] == "fake-information":

                # Sets edge colors
                if (source, currentValue) in self.edge_colors:
                    self.edge_colors[(source, currentValue)] = "orange"
                else:
                    self.edge_colors[(source, currentValue)] = "red"

                # case 1: consumer reject fake info
                if np.random.normal(0.5, .15) > 1 / (1 + math.exp(-currentNode["trustLevel"])):
                    currentNode["trustLevel"] -= .1
                    agent.penalty += 1
                    self.graph.nodes[agent_node]["penalty"] = agent.penalty

                # case 2: consumer accepts fake info
                else:
                    currentNode["trustLevel"] += .1 
                    agent.reward += 1
                    self.graph.nodes[agent_node]["reward"] = agent.reward
                    influenced.add(currentValue)

                    if currentValue in self.node_edge_colors:
                        self.node_edge_colors[currentValue] = "orange"
                    else:
                        self.node_edge_colors[currentValue] = "red"
                    
            elif actionNode["agentType"] == "real-information":

                # Sets edge colors
                if (source, currentValue) in self.edge_colors:
                    self.edge_colors[(source, currentValue)] = "orange"
                else:
                    self.edge_colors[(source, currentValue)] = "blue"

                # case 3: consumer rejects real info
                if np.random.normal(0.5, .15) < 1 / (1 + math.exp(-currentNode["trustLevel"])):
                    currentNode["trustLevel"] += .1
                    agent.penalty += 1
                    self.graph.nodes[agent_node]["penalty"] = agent.penalty
                    
                # case 4: consumer accepts real information
                else:
                    currentNode["trustLevel"] -= .1
                    agent.reward += 1
                    self.graph.nodes[agent_node]["reward"] = agent.reward
                    influenced.add(currentValue)

                    if currentValue in self.node_edge_colors:
                        self.node_edge_colors[currentValue] = "orange"
                    else:
                        self.node_edge_colors[currentValue] = "blue"

            # print(visited, [a for a in self.graph.neighbors(currentValue)])
            for neighbor in self.graph.neighbors(currentValue):
                if currentValue in influenced and neighbor not in visited and neighbor not in queue:
                    queue.append((currentValue, neighbor))
            
            # print('Queue after adding new neighbors:', queue)  


            if currentValue in influenced and currentValue not in agent.influenced_consumers:
                agent.influenced_consumers.append(currentValue)
                currentNode['interactions'].append(agent)


        agent.trustLevel = len(agent.influenced_consumers) / total_nodes
        

        # print(agent.trustLevel)


        # Calculates Rewards/Penalties:
        qVal = self.graph.nodes[agent_node]["qVal"]
        max_qVal = max(agent.reward - agent.penalty, 0)
        self.graph.nodes[agent_node]["qVal"] += 0.1 * (
            agent.reward - agent.penalty + 0.9 * max_qVal - qVal
        )

        return agent.reward, agent.penalty, influenced if influenced else None, qVal


    def step_fact_checker(self, fact_checker_agent, threshold=0.7):
        actions = fact_checker_agent.select_action(threshold=threshold)

        found_fake_info = set()
        for agent_value, action in actions.items():
            if action == 1:
                found_fake = fact_checker_agent.fact_check(agent_value)
               
                if found_fake:
                    found_fake_info.add(agent_value)
        
        print('\nset of fake information agents audited: ', found_fake_info if found_fake_info else None)
        print('Consumer trust level before fact checking ', [c[1]['trustLevel'] for c in self.graph.nodes(data=True)])
        for nodeValue, node_data in self.graph.nodes(data=True):
            if node_data['agentType'] == 'consumer':
                for interaction in node_data['interactions']:
                    if interaction in found_fake_info:
                        node_data['trustLevel'] -= 0.15
        print('Consumer trust level after fact checking', [c[1]['trustLevel'] for c in self.graph.nodes(data=True)])

        # Update the fact-checker's Q-value
        fact_checker_node = self.get_node_from_agent(fact_checker_agent)
        qVal = fact_checker_node["qVal"]
        max_qVal = max(fact_checker_agent.reward - fact_checker_agent.penalty, 0)
        updated_qVal = qVal + 0.1 * (
            fact_checker_agent.reward - fact_checker_agent.penalty + 0.9 * max_qVal - qVal
        )
        fact_checker_node["qVal"] = updated_qVal

        # print(f"Fact-checker Q-value updated: {updated_qVal:.2f}")

    # visualizing the network
    def render(self, mode="human"):
        if not hasattr(self, 'pos'):  
            self.pos = nx.spring_layout(self.graph, seed=42, scale=.2, center=(0, 0))
        
        if mode == "human":
            # print("Graph Nodes and Attributes:")
            # for node, data in self.graph.nodes(data=True):
            #     print(f"Node {node}: {data}")
            # print("Graph Edges:")
            # for src, dst, data in self.graph.edges(data=True):
            #     print(f"Edge {src} -> {dst}: {data}")

            self.drawNetwork()
            self.edge_colors = {}
            self.node_edge_colors = {}


    # Draws the visualization for the network
    def drawNetwork(self):
        node_colors = [
            (
                "blue"
                if self.graph.nodes[node]["agentType"] == "real-information"
                else (
                    "red"
                    if self.graph.nodes[node]["agentType"] == "fake-information"
                    else (
                        "green"
                        if self.graph.nodes[node]["agentType"] == "fact-checker"
                        else "gray"
                    )
                )
            )
            for node in self.graph.nodes
        ]

        edge_colors = [self.edge_colors[edge] if edge in self.edge_colors else "gray" for edge in self.graph.edges()]
        node_edge_colors = [self.node_edge_colors[node] if node in self.node_edge_colors else "gray" for node in self.graph.nodes()]

        plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(
            self.graph, self.pos, node_color=node_colors, edgecolors=node_edge_colors, node_size=500, alpha=0.8
        )
        nx.draw_networkx_edges(self.graph, self.pos, edge_color=edge_colors, alpha=0.5, width=3, arrows=True)
        
        nx.draw_networkx_labels(self.graph, self.pos, labels={node: str(node) for node in self.graph.nodes}, font_size=10)

        plt.scatter([], [], color="blue", label="Real Information Agent")
        plt.scatter([], [], color="red", label="Fake Information Agent")
        plt.scatter([], [], color="green", label="Fact Checker Agent")
        plt.scatter([], [], color="gray", label="Consumer Agent")
        plt.legend(loc="upper right", fontsize=10, bbox_to_anchor=(1.15, 1))
        
        plt.title("Social Network Graph", fontsize=14)
        plt.axis("off")
        plt.show()

    def get_value_from_agent(self, agent):
        return self.agent_to_node_map[agent]
    
    def get_node_from_agent(self, agent):
        return self.graph.nodes[self.get_value_from_agent(agent)]
    

    def print_consumer_info(self):
        """
        Prints trust levels and other information for each consumer node.
        """
        print("Consumer Node Info:")
        for node, data in self.graph.nodes(data=True):
            if data["agentType"] == "consumer":
                trust_level = data["trustLevel"]
                interactions = len(data["interactions"])
                print(f"\nNode {node} - Trust Level: {trust_level:.2f}, Interactions: {interactions}")


    def visualize_rewards_penalties(self, agent, rewards, penalties):
        """
        Visualizes the rewards vs. penalties for a specific agent over time.
        
        Parameters:
        - agent: The agent being tracked (NewsAgent or FactCheckerAgent).
        - rewards: List of rewards accumulated over epochs.
        - penalties: List of penalties accumulated over epochs.
        """
        epochs = range(1, len(rewards) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, rewards, label="Rewards", marker='o')
        plt.plot(epochs, penalties, label="Penalties", marker='o')
        
        plt.title(f"Reward vs Penalty for Agent: {agent.get_type()}", fontsize=16)
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.show()



    def visualize_consumer_trust_levels(self, trust_levels_history):
        """
        Visualizes the trust levels of all consumer nodes over time.

        Parameters:
        - trust_levels_history: A dictionary where keys are consumer node IDs and values are lists of trust levels
        over epochs.
        """
        plt.figure(figsize=(12, 8))
        
        for consumer, trust_levels in trust_levels_history.items():
            epochs = range(1, len(trust_levels) + 1)
            plt.plot(epochs, trust_levels, label=f"Consumer {consumer}", marker='o', alpha=0.7)
        
        plt.title("Consumer Trust Levels Over Epochs", fontsize=16)
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Trust Level", fontsize=14)
        plt.legend(fontsize=10, loc="upper right", bbox_to_anchor=(1.25, 1))
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.show()


    def visualize_agent_influence(self, influence_history):
        """
        Visualizes the influence of all agents in the network based on their trust levels.

        Parameters:
        - influence_history: A dictionary where keys are agent IDs (nodes) and values are lists of trust levels
        over epochs.
        """
        plt.figure(figsize=(12, 8))
        
        for agent, trust_levels in influence_history.items():
            agent_type = agent.get_type()
            epochs = range(1, len(trust_levels) + 1)
            plt.plot(epochs, trust_levels, label=f"{agent_type} (Agent {id(agent)})", marker='o', alpha=0.7)
        
        plt.title("Agent Influence (Trust Level) Over Epochs", fontsize=16)
        plt.xlabel("Epochs", fontsize=14)
        plt.ylabel("Trust Level (Influence)", fontsize=14)
        plt.legend(fontsize=10, loc="upper right", bbox_to_anchor=(1.25, 1))
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.show()


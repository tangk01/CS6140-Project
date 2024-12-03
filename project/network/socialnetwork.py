import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import networkx as nx
import math
from agents.news_agent import NewsAgent
from agents.fact_checker_agent import FactCheckerAgent

class SocialNetworkEnv(gym.Env):
    def __init__(self, numConsumer=10):
        super().__init__()
        self.graph = nx.DiGraph()
        self.numConsumers = numConsumer

        self.build_consumer_network(numConsumer)
        self.build_action_space()
        self.build_obersvation_space()
        

        self.network_size = self.numConsumers

        self.agent_to_node_map = {}
        self.edge_colors = {}


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
            self.graph.add_node(i, agentType="consumer", trustLevel=0.0, storedInfo=[], reward=0, penalty=0)

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
        self.action_space = spaces.Box(low=0, high=1, shape=(self.numConsumers,), dtype=np.int32)
        return self.action_space

    def draw_sample_from_action_space(self):
        '''
        creates the sample space the agent will act upon. 

        i.e; if numConsumer = 4 this will return a vector [0, 1, 1, 0] for example
        where 0 != send, and 1 = send info
        '''
        action = self.action_space.sample()
        return action

    def build_obersvation_space(self):
        self.observation_space = spaces.Dict(
            {
                "trustLevels": spaces.Box(
                    low=0, high=1, shape=(self.numConsumers,), dtype=np.float32
                ),
            }
        )
        return self.observation_space

    # Reset Env
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset agents
        for node in self.graph.nodes:
            nodeType = self.graph.nodes[node]["type"]
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
        
        '''
        Notes for step function trust levels...
        overall network trustlevel 
        trustlevel = .5 (f/r)
        trustlevel -= .1 > .4
        less likely to believe in fake? 

        (node_num, fake-Info)
        1 fake news 
        each consumer get own trustLevel iff multiple news source 
        each news source -> network trust lvl 
        news reliability % (not now)
    if agent not in self.agent_to_node_map:
        raise ValueError("Agetn not found in the network")
        '''     
        agent_node = self.agent_to_node_map[agent]
        actionNode = self.graph.nodes[agent_node]
        visited = set()
        queue = []

        # amt of influence an agent will invoke on the network.
        # agent_node = src node, and trust_in_src_agent = amt of influence agent has in n/w
        total_nodes = self.numConsumers
        print('total_nodes in nw', total_nodes)
        trust_in_src_agent = f"{len(agent.influenced_consumers) / total_nodes:.2f}"
        print("original trust in src agent", trust_in_src_agent)

        # based on agents actions: (1) = send news to nieghbor, (0) dont send news to neighbor.
        for neighbor, sendInfo in zip(self.graph.neighbors(agent_node), action):
            if sendInfo == 1:
                queue.append((agent_node, neighbor))

        while queue:
            info = queue.pop(0)
            print(info)
            curVal = info[1]
            if curVal in visited:
                continue

            visited.add(curVal)
            curNode = self.graph.nodes[curVal]

            
            if curNode["agentType"] == "consumer":

                # Sets edge colors
                if (curVal, info[0]) in self.graph.edges():
                    if (curVal, info[0]) in self.edge_colors:
                        self.edge_colors[(curVal, info[0])] = "orange"
                    else:
                        self.edge_colors[(curVal, info[0])] = "red"
                elif (info[0], curVal) in self.edge_colors:
                    self.edge_colors[(info[0], curVal)] = "orange"
                else:
                    self.edge_colors[(info[0], curVal)] = "red"
                

                # case 1: consumer reject fake info
                if actionNode["agentType"] == "fake-information":
                    if np.random.random() > 1 / (1 + math.exp(-curNode["trustLevel"])):
                        curNode["trustLevel"] = max(0, curNode["trustLevel"] - 0.1)
                        agent.penalty += 1
                        self.graph.nodes[agent_node]["penalty"] = agent.penalty


                    # case 2: consumer accepts fake info
                    else:
                        curNode["trustLevel"] = max(0, curNode["trustLevel"] + 0.1) # increment trustLevel as we will offset this with fact checker
                        agent.reward += 1
                        self.graph.nodes[agent_node]["reward"] = agent.reward


                # case 3: consumer rejects real info
                elif actionNode["agentType"] == "real-information":

                    # Sets edge colors
                    if (curVal, info[0]) in self.graph.edges():
                        if (curVal, info[0]) in self.edge_colors:
                            self.edge_colors[(curVal, info[0])] = "orange"
                        else:
                            self.edge_colors[(curVal, info[0])] = "blue"
                    elif (info[0], curVal) in self.edge_colors:
                        self.edge_colors[(info[0], curVal)] = "orange"
                    else:
                        self.edge_colors[(info[0], curVal)] = "blue"


                    if np.random.random() < 1 / (1 + math.exp(-curNode["trustLevel"])):
                        curNode["trustLevel"] = min(1, curNode["trustLevel"] + 0.1)
                        agent.reward += 1
                        self.graph.nodes[agent_node]["reward"] = agent.reward

                    
                    # case 4: consumer accepts real information
                    else:
                        curNode["trustLevel"] = max(0, curNode["trustLevel"] - 0.1)
                        agent.penalty += 1
                        self.graph.nodes[agent_node]["penalty"] = agent.penalty



                if curVal not in agent.influenced_consumers:
                    agent.influenced_consumers.append(curVal)
                trust_in_src_agent = len(agent.influenced_consumers) / total_nodes
                curNode["storedInfo"].append((agent_node, f"{trust_in_src_agent:.2f}"))


            for neighbor in self.graph.neighbors(curVal):
                if neighbor not in visited:
                    queue.append((curVal, neighbor))

        print('agent', agent_node)
        print('num of influenced consumer from agent ', agent_node, len(agent.influenced_consumers))
        print('all consumers', agent_node, ' influenced', agent.influenced_consumers)
        print('total trust_in src agent', trust_in_src_agent)


        # Calculates Rewards/Penalties:
        qVal = self.graph.nodes[agent_node]["qVal"]
        max_qVal = max(agent.reward - agent.penalty, 0)
        self.graph.nodes[agent_node]["qVal"] += 0.1 * (
            agent.reward - agent.penalty + 0.9 * max_qVal - qVal
        )

        # Return the updated state
        agent.trustLevels = np.array(
            [self.graph.nodes[i]["trustLevel"] for i in range(self.numConsumers)]
        )
        info = {}

        return {"trustLevels": agent.trustLevels}, agent.reward, info

    # visualizing the network
    def render(self, mode="human"):
        if not hasattr(self, 'pos'):  
            self.pos = nx.spring_layout(self.graph, seed=42, scale=.2, center=(0, 0))
        
        if mode == "human":
            print("Graph Nodes and Attributes:")
            for node, data in self.graph.nodes(data=True):
                print(f"Node {node}: {data}")
            print("Graph Edges:")
            for src, dst, data in self.graph.edges(data=True):
                print(f"Edge {src} -> {dst}: {data}")

        self.drawNetwork()
        self.edge_colors = {}

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

        plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(
            self.graph, self.pos, node_color=node_colors, node_size=500, alpha=0.8
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
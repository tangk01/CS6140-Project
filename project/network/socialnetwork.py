import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import networkx as nx

class SocialNetworkEnv(gym.Env):
    def __init__(self, numConsumer = 10):
        super().__init__()
        agent_type = np.random.choice(["real-information", "fake-information", "fact-checker", "consumer"])
        self.graph = nx.DiGraph()

        for i in range(numConsumer):
            self.graph.add_node(i,
                                type="consumer",
                                trustLevel=0.0, 
                                storedInfo=[], 
                                reward=0, 
                                penalty=0)
        
        # Each consumer connects with ~2 others
        for _ in range(numConsumer * 2):  
            src = np.random.randint(0, numConsumer)
            dst = np.random.randint(0, numConsumer)
            if src != dst:
                self.graph.add_edge(src, dst, weight=1.0)


        # Fake information agent connected with every consumer
        self.graph.add_node(numConsumer,
                            type="fake-information", 
                            qVal=0.0, 
                            reward=0, 
                            penalty=0)
        
        for node in self.graph.nodes:
            if node != numConsumer:
                self.graph.add_edge(numConsumer, node)

        

        self.numConsumers = numConsumer
        #Action Space is a binary array indicating whether the agent sends information
        self.action_space = spaces.Box(low=0, high=1, shape=(numConsumer,), dtype=np.int32)
        print(self.action_space)
        self.observation_space = spaces.Dict({
            "trustLevels": spaces.Box(low=0, high=1, shape=(numConsumer,), dtype=np.float32),
        })


    #Reset Env
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
        trustLevels = np.array([self.graph.nodes[i]["trustLevel"] for i in range(self.numConsumers)])
        
        # Return initial observation
        return {"trustLevels": trustLevels}, {}

    #Step Function for Env
    #NEED TO FIX
    def step(self, action, agent = 10):
        print(action)
        rewards = 0
        penalties = 0

        #CHANGE LATER
        actionNode = self.graph.nodes[agent]

        visited = set()
        queue = []

        #Goes through and spreads news from source
        for neighbor, sendInfo in zip(self.graph.neighbors(agent), action):
            if sendInfo == 1:
                queue.append(neighbor)

        while queue:
            curVal = queue.pop()
            if curVal in visited:
                break

            visited.add(curVal)
            curNode = self.graph.nodes[curVal]

            if curNode["type"] == "consumer":
                # Update trust-level and stored-information based on the source
                if actionNode["type"] == "fake-information" and np.random.random() > 1/(1 + math.exp(-curNode["trustLevel"])):
                    curNode["trustLevel"] -= 0.1
                    rewards += 1

                    for neighbor in self.graph.neighbors(curVal):
                        queue.append(neighbor)
                elif actionNode["type"] == "real-information" and np.random.random() < 1/(1 + math.exp(-curNode["trustLevel"])):
                    curNode["trustLevel"] += 0.1
                    rewards += 1

                    for neighbor in self.graph.neighbors(curVal):
                        queue.append(neighbor)


        #Calculates Rewards/Penalties
        #CHANGE LATER
        max_qVal = max(rewards - penalties, 0)
        self.graph.nodes[agent]["qVal"] += 0.1 * (rewards - penalties + 0.9 * max_qVal - self.graph.nodes[agent]["qVal"])
    
        # Return the updated state
        trustLevels = np.array([self.graph.nodes[i]["trustLevel"] for i in range(self.numConsumers)])
        done = False  # In this simulation, the environment does not end
        info = {}

        return {"trustLevels": trustLevels}, rewards, done, info
    
    #Render the graph for debugging or visualization
    #NEED FIX
    def render(self, mode="human"):
        if mode == "human":
            print("Graph Nodes and Attributes:")
            for node, data in self.graph.nodes(data=True):
                print(f"Node {node}: {data}")
            print("Graph Edges:")
            for src, dst, data in self.graph.edges(data=True):
                print(f"Edge {src} -> {dst}: {data}")


            pos = nx.spring_layout(self.graph)
            node_colors = [
                    "blue" if self.graph.nodes[node]["type"] == "real-information"
                    else "red" if self.graph.nodes[node]["type"] == "fake-information"
                    else "green" if self.graph.nodes[node]["type"] == "fact-checker"
                    else "gray"
                    for node in self.graph.nodes
                ]
            
            plt.figure(figsize=(8,8))
            nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=500, alpha=0.8)
            nx.draw_networkx_edges(self.graph, pos, alpha=0.5, arrows=True)
            # nx.draw(self.graph, layout=nx.spring_layout(self.graph))

            plt.title("Social Network Graph", fontsize=14)
            plt.axis("off")
            plt.show()
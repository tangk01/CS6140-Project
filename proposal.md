# CS6140 Final Project

<li> Kevin Tang - tang.kevi@northeastern.edu </li>
<li> Chase Coogan - coogan.c@northeastern.edu </li>
<br>

This project aims to model and analyze the evolution of opinions within a social network when influenced by false information. We will leverage reinforcement learning to train agents to resemble the general public, malicious falsehood spreaders, and fact-checkers to understand the role each plays in facilitating opinion dynamics. By simulating these social interactions, the underlying impact of accurate and misleading information can be revealed. This is significant as it addresses critical challenges in today's digital age, where false information can rapidly spread across social media platforms, impacting both public opinion and decision-making processes.

Our motivation stems from applying reinforcement learning to a pressing real-world issue. The social discourse and political polarization from false information could have a detrimental effect on real-world decisions. As a result, we hope our project will provide some insight on how fact-checkers can effectively counter false information and influence opinion dynamics. This in turn will result in better algorithms or policies for combating the spread of false information.

Opinion dynamics in social networks is essentially the study of how an individual’s opinion forms, evolves, and spreads over time within an interconnected network of other individuals. In the digital age, all social media is a form of a social network, which facilitates an exchange of information for opinion making. This information can be classified into one of two categories, truths or falsehoods. Here falsehoods include misinformation and disinformation, which both mislead the public and complicate decision making. To simulate this information exchange, we have decided to use a reinforcement learning approach. This approach will allow us the ability to create various agents, including the general public (who form and evolve opinions based on interactions), malicious falsehood spreaders (who spread misinformation or disinformation), and fact-checking agents (who attempt to counteract falsehoods). These agents will then interact and adapt to the changing environment, allowing for realistic simulations of how opinions spread. This will then hopefully reveal existing and hidden concepts in social influence theory like confirmation bias (favoring information that aligns with one’s preexisting beliefs) and echo chambers (groups where similar opinions reinforce each other, often excluding contrary views).

Research on opinion dynamics has been done previously and several different models have been created to better understand how these opinions spread and update. One of these models is the DeGroot model, where individuals update their beliefs based on the opinions of their neighbors until a consensus is reached. Similarly, the Hegselmann-Krause model considers agents that only interact with those whose opinions are within a certain range of their own, thereby simulating opinion polarization. Additionally, this topic does not solely stem from computer science and various fields like sociology and psychology have investigated to create a comprehensive model.

Our project, however, is distinct from the others in a few regards. The first, is by using a reinforcement learning model we can have a dynamic and ever evolving system which more closely mimics the interactions in the real world. By leveraging the malicious and benevolent agents, each group will adapt and find strategies to complete their own specific objectives. From here lies the second distinct element which is how we are incorporating both misinformation spreaders and fact-checkers. Most models focus on the spread of information between individuals based on their current opinions, but we wanted to focus on the impact of a third party. By introducing a separate entity in charge of spreading information, we can simulate the effect corporations and social media have on the public. This way we can create algorithms and policies to regulate the spread of false information to the public.

# section e

We will be generating our own data for this experiment as our project will rely heavily on simulations where agents invoke a set of actions and socialize across our network. As they spread information, there attributes will be updated. For example, as a fake-news agent spreads misinformation and it tricks an agent into believing it its trust-level will increase which will affect the next iteration as it now has more influence in the network.

In order to carry out this experiment, we plan to create a social network using a graph structure where we will have various agents representing real-news, fake-news, fact-checkers and regular agents. Real-news agents are responsible for sharing factual based news. Fake-news will do the opposite, spreading fake information. Fact-checkers will verify both real and fake news that gets sent across the network, and regular agents will be a consumer of all types of news. These agents will be the nodes in our graph, where edges represent a social relationship. With this type of envirionment, we can construct graph networks that allow us to change how many of each type of agent we want introduced to the network to see how a differing size of agents may change the outcome. Agents will "communicate" with each other by passing news around where each agent will have to decide to trust or not trust the source.

Agents will have a type 'real-news, fake-news, fact-checker, regular' that define what their mission is. Agents will be rewarded or penalized based on the amount of influence they create across the network. real-news and fake-news agents are responsible for spreading their news where they will be rewarded for the amount of influence they invoke. Fact-checkers will have the responsibility to check the legimitacy behind the news being spread. A fact-checker will be rewarded for correctly identifying fake-news sources and will be penalized for incorrectly labeling real-news as fake. Fake-news agents will be penalized when fact-checkers catch them.

All agents will store the news passed to them, and the agent that sent it. They will also store the overall trust level in that news. Each agent will store its reward and penalty amount to help train them which will be dependent on either the number of agents influenced, or number of news articles correctly debunked.

In order to evaluate how our agents are performing, we will invoke Q-Learning to help train them properly based on reward/penalty.The update rule for Q-Learning that we can use is Bellmans equation $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma max Q(s`, a`) - Q(s, a) \right]$$

$s$ = current state  
$a$ = action taken in state s  
$r$ = reward received after taking action a  
$s'$ = next state resulting from action a  
$α$ = learning rate  
$γ$ = discount factor (importance of future rewards vs immediate rewards)

Each agent will need to act independently to update its Q-value based on its interactions with other agents. Each agent will also need to hold a set of actions it can take which we can define for each agent:

Fake-news agent:

1. spread misinformation
2. do nothing

Real-news agent:

1. spread truthful information
2. do nothing

Fact-checking agent:

1. investigate the message/agent and label it real/fake
2. ignore the message (do nothing)

Regular agent:

1. believe and share a message
2. do not believe message/do not share

We can use the epsilon-greedy strategy to balance exploration to exploitation. Meaning, we will assign some probability $\epsilon$ as the probability the agent chooses a random action (exploration). And, we can assign 1 - $\epsilon$ as the probability the agent chooses the action with the highest Q-value. After an agent performs an action they will go through a state change that increases or decreases their influence in the network.

We expect that the fake-news agents to try to learn strategies to act in its favor to spread as much misinformation as possible, and possibly try to avoid being detected by the fact-checkers. We suspect they will try to target agents that have low skepticism. We also epect echo-chambers to form amongst the network where misinformation is being shared continously. We expect the real-news agents to act similarly, where their main goal is to maximize the amount of real information they spread. We expect the real-news agents to focus on areas where fact-checkers are more dominant. We expect fact-checkers to invoke an optimal strategy for uncovering fake-news. We are hoping that fact-checkers are able to effectively differentiate between real and fake news such that it is able to correctly label them. If we run into any issues where failure may occur, we plan to reduce the number of actions that each agent can take, and possible reduce the nubmer of types of agents we use. We will also look into how to optimally reward our agents so that they behave how we have expected them to.

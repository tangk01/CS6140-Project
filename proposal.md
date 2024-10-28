Title: 

  

Kevin Tang - tang.kevi@northeastern.edu 

Chase Coogan - coogan.c@northeastern.edu 

  

This project aims to model and analyze the evolution of opinions within a social network when influenced by false information. We will leverage reinforcement learning to train agents to resemble the general public, malicious falsehood spreaders, and fact-checkers to understand the role each plays in facilitating opinion dynamics. By simulating these social interactions, the underlying impact of accurate and misleading information can be revealed. This is significant as it addresses critical challenges in today's digital age, where false information can rapidly spread across social media platforms, impacting both public opinion and decision-making processes. 

Our motivation stems from applying reinforcement learning to a pressing real-world issue. The social discourse and political polarization from false information could have a detrimental effect on real-world decisions. As a result, we hope our project will provide some insight on how fact-checkers can effectively counter false information and influence opinion dynamics. This in turn will result in better algorithms or policies for combating the spread of false information. 

 

Opinion dynamics in social networks is essentially the study of how an individual’s opinion forms, evolves, and spreads over time within an interconnected network of other individuals. In the digital age, all social media is a form of a social network, which facilitates an exchange of information for opinion making. This information can be classified into one of two categories, truths or falsehoods. Here falsehoods include misinformation and disinformation, which both mislead the public and complicate decision making. To simulate this information exchange, we have decided to use a reinforcement learning approach. This approach will allow us the ability to create various agents, including the general public (who form and evolve opinions based on interactions), malicious falsehood spreaders (who spread misinformation or disinformation), and fact-checking agents (who attempt to counteract falsehoods). These agents will then interact and adapt to the changing environment, allowing for realistic simulations of how opinions spread. This will then hopefully reveal existing and hidden concepts in social influence theory like confirmation bias (favoring information that aligns with one’s preexisting beliefs) and echo chambers (groups where similar opinions reinforce each other, often excluding contrary views). 

Research on opinion dynamics has been done previously and several different models have been created to better understand how these opinions spread and update. One of these models is the DeGroot model, where individuals update their beliefs based on the opinions of their neighbors until a consensus is reached. Similarly, the Hegselmann-Krause model considers agents that only interact with those whose opinions are within a certain range of their own, thereby simulating opinion polarization. Additionally, this topic does not solely stem from computer science and various fields like sociology and psychology have investigated to create a comprehensive model. 

Our project, however, is distinct from the others in a few regards. The first, is by using a reinforcement learning model we can have a dynamic and ever evolving system which more closely mimics the interactions in the real world. By leveraging the malicious and benevolent agents, each group will adapt and find strategies to complete their own specific objectives. From here lies the second distinct element which is how we are incorporating both misinformation spreaders and fact-checkers. Most models focus on the spread of information between individuals based on their current opinions, but we wanted to focus on the impact of a third party. By introducing a separate entity in charge of spreading information, we can simulate the effect corporations and social media have on the public. This way we can create algorithms and policies to regulate the spread of false information to the public. 

# gail-carracing!

Everytime forward Reinforcement Learning(RL) is not feasible for all of the problems due to the complexity involved in the designing of the reward function.
In those circumstances, Inverse Reinforcement Learning(IRL) is the game changer. Imitation learning technique is part of it and it showed wonderful results on some of the problems.

In this project, I created an agent that tries to imitate the expert and learns the path navigation in the process. Thanks to openAI-Gym simulator for providing such a wonderful platform for creating the dynamics of the environment.

The project is divided into two steps
1. Triaining the expert using Proximal Policy Optimization(PPO) algorithm
2. Train the agent using the expert trajectories from the step1 by utilizing GAN architecture.

Design:

![alt text](https://user-images.githubusercontent.com/32699857/147871510-b1927865-e57f-4d7a-9693-839310b01b3e.PNG)

Results:
![alt text](https://user-images.githubusercontent.com/32699857/147870992-7873ad00-2ac7-4cb5-8f9c-d5bcb883d4ed.gif)

More details can be found in the report.


# El Farol Bar Reinforcement Learning

### Description
This project implements a reinforcement learning agent using the Deep Deterministic Policy Gradient (DDPG) algorithm to solve the El Farol bar problem. The objective is to develop an agent capable of deciding whether to go to the bar based on expected crowds.

### Acknowledgments
This implementation was inspired by Duy Minh Le's DDPG code on Kaggle (www.kaggle.com/code/duyminhle/deep-deterministic-policy-gradients-ddpg). Additionally, the other scripts in this project were inspired by Mr. Muller's implementation of the Tic Tac Toe game, as well as codes from my colleagues that were shared in the Moodle forum. ChatGPT also provided some assistance in developing the trainer.py script.

### Project structure
ddpg.py: Implements the DDPG agent.  
elfarol_env.py: Defines the El Farol bar environment.  
trainer.py: Script for training the agent in the El Farol environment.

### This code works ! You can try it.
- To run this project, you will need Python and a few specific libraries. You can install the dependencies via pip:  
pip install tensorflow numpy gymnasium pettingzoo matplotlib

- To train the agent, simply run the trainer.py script:  
python trainer.py

This script will train the agent over a set number of episodes and display the agent's performance in terms of the average number of people present at the bar.

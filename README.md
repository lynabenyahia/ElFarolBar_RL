# El Farol Bar Reinforcement Learning

### Description
This project implements a reinforcement learning agent using the Deep Deterministic Policy Gradient (DDPG) algorithm to solve the El Farol bar problem. The objective is to develop an agent capable of deciding whether to go to the bar based on expected crowds.

### Project structure
ddpg.py: Implements the DDPG agent.
elfarol_env.py: Defines the El Farol bar environment.
trainer.py: Script for training the agent in the El Farol environment.

### Installation
- To run this project, you will need Python and a few specific libraries. You can install the dependencies via pip:
pip install tensorflow numpy gymnasium pettingzoo matplotlib

- To train the agent, simply run the trainer.py script:
python trainer.py

This script will train the agent over a set number of episodes and display the agent's performance in terms of the average number of people present at the bar.

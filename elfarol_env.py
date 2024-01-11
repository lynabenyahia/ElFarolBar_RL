from typing import Optional
import numpy as np
from gymnasium.spaces import Discrete
from pettingzoo import ParallelEnv

class elfarol_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self,
                 N: Optional[int]=10, # number of agents - potential visitors at the bar
                 optimal_crowd: Optional[int]=6, # number of people for best enjoying the bar
                 render_mode=None,
                 max_steps=50):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        These attributes should not be changed after initialization.
        """
        self.N = N # number of potential agents
        self.possible_agents = ["player_" + str(r) for r in range(N)]
        self.optimal_crowd = optimal_crowd
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        # state of the bar - how crowded is it? If it's still closed - None
        self.state = None
        # parameters for the reward function
        self.b = 1
        self.a = self.optimal_crowd*2*self.b # to have optimum at optimal crowd

    # Observation space - each agent sees the same - does not depend on agent
    # Observation - Number of people in the bar
    def observation_space(self,
                          agent: Optional[str]=None):
        return Discrete(1)

    # Action space - go-1 or don't go-0
    def action_space(self,
                     agent: Optional[str]=None):
        return Discrete(1)

    def render(self):
        print(self.state)

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        self.current_step = 0
        observations = {agent: 0 for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        self.state = None

        return observations, infos

    def _calc_rewards(self, actions):
        """
        calculates reward for each agent, as fct. of agent-decision and number of agents going
        actions - should be np.array (with zero's and one's)
        :return:
        reward - np.array - same length as action vector
        """
        rewards = np.zeros(len(actions))  # Créez un array de zéros de la taille du nombre d'agents
        crowd = 0
        
        for i, (agent_id, action_array) in enumerate(actions.items()):
            action = action_array[0]  # Prendre la première valeur de l'action array
            crowd += action  # Ajouter cette action au total de la foule
    
            if action == 0:
                rewards[i] = 0
            elif crowd < 0.2 * self.N:
                rewards[i] = -10
            elif crowd < 0.5*self.N:
                rewards[i] = -5
            elif crowd < 0.7*self.N:
                rewards[i] = 10
            elif crowd < 0.9*self.N:
                rewards[i] = 1
            else:
                rewards[i] = -5

        return rewards, crowd


    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        self.current_step += 1
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
    
        # Rewards for every agents are placed in a dict of rewards to return 
        rewards, crowd = self._calc_rewards(actions)
        rewards = {agent_id: rewards[i] for i, agent_id in enumerate(self.agents)}
    
        # Check if the maximal number of steps is reach 
        if self.current_step >= self.max_steps:
            terminations = {agent: True for agent in self.agents}
            truncations = {agent: True for agent in self.agents}
            self.current_step = 0
        else:
            terminations = {agent: False for agent in self.agents}
        
        self.num_moves += 1
        env_truncation = False
        truncations = {agent: env_truncation for agent in self.agents}
    
        self.state = crowd
        observations = {agent: self.state for agent in self.agents}  # update the observations for each agent
    
        infos = {agent: {} for agent in self.agents}
    
        if env_truncation:
            self.agents = []
    
        if self.render_mode == "human":
            self.render()
    
        return observations, rewards, terminations, truncations, infos

import numpy as np
from ddpg import Agent  
from elfarol_env import elfarol_env 
import matplotlib.pyplot as plt 

def main():
    """
    Main function to run the El Farol bar problem simulation.
    """
    num_episodes = 5000  
    env = elfarol_env(N=10, optimal_crowd=6)  # Initialize the El Farol environment
    agent = Agent(input_dims=[1], alpha=0.01, beta=0.01, tau=0.5)  # Initialize the agent
    average_people = []  # To store the average number of people in the bar per episode

    for episode in range(num_episodes):
        state_dict, _ = env.reset()  
        done = False  # Flag to indicate the end of an episode
        people_count = [] 

        while not done:
            actions = {}  
            for agent_id in env.agents:
                observation = state_dict[agent_id]  
                observation = np.array([observation])  
                action_prob = agent.choose_action(observation)  
                action = np.random.binomial(1, action_prob)  # Sample an action based on the probability
                actions[agent_id] = action  

            next_state_dict, rewards, dones, _, _ = env.step(actions)  
            done = any(dones.values()) 

            people_at_bar = sum(actions.values()) 
            people_count.append(people_at_bar)

            for agent_id in env.agents:
                obs = state_dict[agent_id]  
                new_obs = next_state_dict[agent_id] 
                reward = rewards[agent_id]  
                done = dones[agent_id]  # Whether the episode is done for the agent
                obs = np.array([obs])  
                new_obs = np.array([new_obs])  
                agent.remember(obs, actions.get(agent_id, 0), reward, new_obs, done)  # Store the transition
                agent.learn()  # Agent learns from the transition

            state_dict = next_state_dict  # Update the state dictionary for the next step

        average_people_this_episode = np.mean(people_count)  
        average_people.append(average_people_this_episode) 
        
        print(f"Episode {episode}")
        print(f"Average number of people at the bar : {average_people_this_episode}")

    # Plotting the graph
    plt.plot(average_people, color='r') 
    plt.title("Average number of people per episode")  
    plt.xlabel("Episodes") 
    plt.ylabel("Average number of people at the bar")
    plt.axhline(y=6, color='black', linestyle='-', linewidth=1)  # Draw a line at y=6 (optimal crowd)
    plt.ylim(0, 10)  
    plt.show() 

if __name__ == "__main__":
    main() 

import tensorflow as tf  
import numpy as np  

class ReplayBuffer: 
    def __init__(self, max_size, input_shape, n_actions):
        """
        A class used to store and sample transitions for reinforcement learning.
        """
        self.mem_size = max_size  
        self.mem_cntr = 0  # Initialize a memory counter 
        
        # Initialize memory arrays for states, new states, actions, rewards, and terminal flags
        self.state_memory = np.zeros((self.mem_size, *input_shape))  # Memory for storing states
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))  # Memory for storing new states after action
        self.action_memory = np.zeros((self.mem_size, n_actions)) 
        self.reward_memory = np.zeros(self.mem_size)  
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_) 

    # Function to store a transition in the buffer
    def store_transition(self, state, action, reward, state_, done):
        """
        Stores a transition in the replay buffer.
        """
        index = self.mem_cntr % self.mem_size  # Calculate index for storing the new transition

        # Store the given transition at the calculated index
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
    
        self.mem_cntr += 1  

    # Function to sample a batch of transitions from the buffer
    def sample_buffer(self, batch_size):
        """
        Samples a batch of transitions from the buffer.

        Parameters:
            batch_size (int): The size of the batch to sample.

        Returns:
            tuple: A batch of (states, actions, rewards, next_states, dones).
        """
        max_mem = min(self.mem_cntr, self.mem_size)  # Determine the current size of the buffer
    
        batch = np.random.choice(max_mem, batch_size, replace=False)  # Randomly select a batch of indices
        # Extract the sampled transitions using the selected indices
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
    
        return states, actions, rewards, states_, dones 

# Define a Critic Network class for value estimation
class CriticNetwork(tf.Module):
    """
    A neural network class for estimating Q-values in reinforcement learning.
    """
    def __init__(self, input_dims, fc1_dims=64, n_actions=1, name='critic'):
        super(CriticNetwork, self).__init__(name=name)  # Initialize the base class
        self.fc1_dims = fc1_dims  

        # Initialize weights for the first fully connected layer and Q-value output layer
        self.fc1_weights = tf.Variable(tf.random.normal([input_dims[0] + n_actions, self.fc1_dims]), trainable=True)
        self.q_weights = tf.Variable(tf.random.normal([self.fc1_dims, 1]), trainable=True)

    # Function to compute Q-value given state and action
    def __call__(self, state, action):
        """
        Computes the Q-value for a given state and action.

        Parameters:
            state (tf.Tensor): The current state.
            action (tf.Tensor): The action taken.

        Returns:
            tf.Tensor: The estimated Q-value.
        """
        state_action = tf.concat([state, action], axis=1)  
        action_value = tf.matmul(state_action, self.fc1_weights)  # Compute the first layer output
        action_value = tf.nn.relu(action_value)  # Apply ReLU activation

        q = tf.matmul(action_value, self.q_weights)  # Compute the Q-value

        return q 

# Define an Actor Network class for policy approximation
class ActorNetwork(tf.Module):
    """
    A neural network class for generating action policies in reinforcement learning.
    """
    def __init__(self, input_dims, fc1_dims=64, n_actions=1, name='actor'):
        super(ActorNetwork, self).__init__(name=name)  # Initialize the base class
        self.fc1_dims = fc1_dims 

        # Initialize weights for the first fully connected layer and policy output layer
        self.fc1_weights = tf.Variable(tf.random.normal([input_dims[0], self.fc1_dims]), trainable=True)
        self.mu_weights = tf.Variable(tf.random.normal([self.fc1_dims, n_actions]), trainable=True)

    # Function to compute the policy output given a state
    def __call__(self, state):
        """
        Computes the policy output for a given state.

        Parameters:
            state (tf.Tensor): The current state.

        Returns:
            tf.Tensor: The actions as probabilities according to the learned policy.
        """
        prob = tf.matmul(state, self.fc1_weights)  # Compute the first layer output
        prob = tf.nn.relu(prob)  # Apply ReLU activation

        mu = tf.matmul(prob, self.mu_weights)  # Compute the policy output
        mu = tf.nn.tanh(mu)  # Apply tanh activation for policy output, to scale actions

        return mu 

# Define an Agent class for the reinforcement learning model
class Agent:
    """
    A class representing an agent for reinforcement learning, using Actor-Critic networks.
    """
    def __init__(self, alpha, beta, input_dims, tau, gamma=0.99, n_actions=2, max_size=10000, layer1_size=128, batch_size=64, noise=0.1):
        self.gamma = gamma  # Set the discount factor for future rewards
        self.tau = tau  # Set the soft update parameter for target network updates
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)  # Initialize the replay buffer
        self.batch_size = batch_size 
        # Initialize actor and critic networks along with their target networks
        self.actor = ActorNetwork(input_dims, layer1_size, n_actions)
        self.critic = CriticNetwork(input_dims, layer1_size, n_actions)
        self.target_actor = ActorNetwork(input_dims, layer1_size, n_actions)
        self.target_critic = CriticNetwork(input_dims, layer1_size, n_actions)
        self.n_actions = n_actions  
        self.noise = noise 
        # Initialize optimizers for actor and critic networks
        self.actor.optimizer = tf.optimizers.legacy.Adam(learning_rate=alpha)
        self.critic.optimizer = tf.optimizers.legacy.Adam(learning_rate=beta)
        self.min_action = 0  
        self.max_action = 1
        # Perform an initial update of target network parameters
        self.update_network_parameters(tau=1)

    # Function to update network parameters, especially for target networks
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau  

        actor_weights = self.actor.trainable_variables
        critic_weights = self.critic.trainable_variables
        target_actor_weights = self.target_actor.trainable_variables
        target_critic_weights = self.target_critic.trainable_variables

        # Soft update target actor network weights
        for i in range(len(actor_weights)):
            target_actor_weights[i].assign(tau * actor_weights[i] + (1 - tau) * target_actor_weights[i])
        
        # Soft update target critic network weights
        for i in range(len(critic_weights)):
            target_critic_weights[i].assign(tau * critic_weights[i] + (1 - tau) * target_critic_weights[i])

    # Function to store transitions in the replay buffer
    def remember(self, state, action, reward, new_state, done):
        """
        Stores a transition in the agent's replay buffer.
        """
        self.memory.store_transition(state, action, reward, new_state, done)

    # Function to select an action based on current observation
    def choose_action(self, observation, evaluate=False):
        """
        Selects an action based on the current observation.

        Parameters:
            observation (numpy.ndarray): The current observed state.
            evaluate (bool): If True, choose action without adding noise (for evaluation).

        Returns:
            numpy.ndarray: The chosen action.
        """
        state = tf.convert_to_tensor([observation], dtype=tf.float32)  # Convert observation to tensor
        actions = self.actor(state)  
        if not evaluate:  
            actions += tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=self.noise)
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)  # Clip actions to valid range
        actions = tf.sigmoid(actions)  # Apply sigmoid to actions for output mapping
        return actions[0] 

    # Function for the agent to learn from experiences
    def learn(self):
        """
        Performs a learning step using a batch of sampled transitions.

        Updates the actor and critic networks based on the learning from the experiences.
        """
        if self.memory.mem_cntr < self.batch_size:
            return  # Do not learn if not enough samples in the replay buffer

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        # Convert the sampled batch to tensors
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
    
        # Calculate critic loss and update critic network
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_) 
            # Compute critic value for next state and target actions
            critic_value_ = tf.squeeze(self.target_critic(states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)  # Current critic value
            # Compute target value for critic update
            target = rewards + self.gamma * critic_value_ * (1 - done)  # Compute the target Q-value
            critic_loss = tf.reduce_mean(tf.square(target - critic_value))  # Calculate critic loss as mean squared error

        # Compute gradients and update critic network parameters
        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        # Calculate actor loss and update actor network
        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)  # Get new actions from the actor network
            # Calculate actor loss as the negative mean of the critic values (for policy gradient)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.reduce_mean(actor_loss)

        # Compute gradients and update actor network parameters
        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

        # Soft update target networks with the main networks
        self.update_network_parameters()

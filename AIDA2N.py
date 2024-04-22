import numpy as np

# Define environment
num_states = 16
num_actions = 4
q_table = np.zeros((num_states, num_actions))

# Define hyperparameters
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1
num_episodes = 1000

# Define grid world transition function
def transition(state, action):
    next_state = state
    if action == 0:  # Up
        next_state -= 4
    elif action == 1:  # Down
        next_state += 4
    elif action == 2:  # Left
        next_state -= 1
    elif action == 3:  # Right
        next_state += 1
    
    next_state = max(0, min(num_states - 1, next_state))  # Boundaries
    return next_state

# Q-learning algorithm
for episode in range(num_episodes):
    state = np.random.randint(0, num_states)  # Random initial state
    
    while True:
        # Epsilon-greedy policy for action selection
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(0, num_actions)  # Exploration
        else:
            action = np.argmax(q_table[state])  # Exploitation
        
        next_state = transition(state, action)
        reward = -1 if next_state != num_states - 1 else 0  # Reward
        
        # Update Q-value using Q-learning equation
        q_table[state, action] += learning_rate * (reward + 
                            discount_factor * np.max(q_table[next_state]) - q_table[state, action])
        
        state = next_state
        
        if state == num_states - 1:  # Terminal state
            break

# Test the learned policy
state = 0  # Start from initial state
while True:
    action = np.argmax(q_table[state])
    next_state = transition(state, action)
    print(f"State: {state}, Action: {action}")
    state = next_state
    if state == num_states - 1:  # Terminal state
        print("Goal reached!")
        break

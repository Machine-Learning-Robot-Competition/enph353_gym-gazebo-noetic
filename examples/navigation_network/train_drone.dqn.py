import numpy as np
from deepQ import DeepQ  # Import your DeepQ class
from gym_gazebo.envs.navigation_network.navigation_training_env import NavigationTrainingEnv
import rospy 

# Hyperparameters
input_size = 3  # Observation size: [x, y, z]
output_size = 8  # Number of discrete actions
memory_size = 10000
discount_factor = 0.99
learning_rate = 0.001
learn_start = 128
episodes = 1000
max_steps = 20
saveFrequency = 50
exploration_rate = 1.0
exploration_decay = 0.995
min_exploration_rate = 0.01
batch_size = 64

# Initialize the environment and DeepQ network
env = NavigationTrainingEnv()
dqn = DeepQ(input_size, output_size, memory_size, discount_factor, learning_rate, learn_start)
dqn.initNetworks(hiddenLayers=[8, 8])  # Two hidden layers with 64 neurons each

# Training loop
for episode in range(episodes):
    state = env.reset()  # Reset the environment
    total_reward = 0

    for step in range(max_steps):
        print(f"Step {step}: Calling getQValues")
        q_values = dqn.getQValues(state)
        print(f"Q-values: {q_values}")

        print("Selecting action...")
        action = dqn.selectAction(q_values, exploration_rate)
        print(f"Action selected: {action}")

        print("Taking step in the environment...")
        next_state, reward, done, _ = env.step(action)
        rospy.sleep(0.08)  # Allow the environment to process the action
        print(f"Step taken. Next state: {next_state}, Reward: {reward}, Done: {done}")

        dqn.addMemory(state, action, reward, next_state, done)
        
        if step % 5 == 0:  # Train every 5 steps
             dqn.learnOnMiniBatch(batch_size)

        # Update state after next_state is defined
        state = next_state

        total_reward += reward
        if step % saveFrequency == 0:
            dqn.saveModel("dqn_drone_navigation_model.h5")

        if done:
            break


    # Update the exploration rate
    exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)

    # Update the target network periodically
    if episode % 10 == 0:
        dqn.updateTargetNetwork()

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Save the trained model
dqn.saveModel("dqn_drone_navigation_model.h5")
env.close()

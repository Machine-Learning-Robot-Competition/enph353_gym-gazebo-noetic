import numpy as np
from deepQ import DeepQ  # Import your DeepQ class
from gym_gazebo.envs.navigation_network.navigation_training_env import NavigationTrainingEnv
import rospy 
from liveplot import plot_best_actions, plot_rewards_live
import sys
import random


# Hyperparameters
input_size = 4  # Observation size: [x, y, cos(theta). sin(theta)]
output_size = 3  # Number of discrete actions
memory_size = 10000
discount_factor = 0.99
learning_rate = 0.005
learn_start = 128
episodes = 1000
max_steps = 60
exploration_rate = 1.0
# exploration_decay = 0.998
exploration_decay = 0.999
min_exploration_rate = 0.01
batch_size = 64

# for visualizing the q values
# state_space_x = np.linspace(-20, 20, 100)
# state_space_y = np.linspace(-20, 20, 100)
reward_list = [] # for tracking reward progress
# state_arr = np.zeros((max_steps, input_size))
# q_value_arr = np.zeros((max_steps, output_size))

# Initialize the environment and DeepQ network
env = NavigationTrainingEnv()
dqn = DeepQ(input_size, output_size, memory_size, discount_factor, learning_rate, learn_start)
dqn.initNetworks(hiddenLayers=[128, 64, 32, 16])  # Two hidden layers with 64 neurons each

model_path = "dqn_drone_navigation_model.h5"
try:
    dqn.loadWeights(model_path)
    print(f"Model weights successfully loaded from {model_path}")
except Exception as e:
    print(f"Failed to load model weights from {model_path}. Error: {e}")
    # sys.exit(1)

rospy.sleep(10) # wait for other nodes to launch

# Training loop
for episode in range(episodes):
    state = env.reset()  # Reset the environment
    rospy.sleep(1)
    total_reward = 0
    

    for step in range(max_steps):
        # print(f"Step {step}: Calling getQValues")
        q_values = dqn.getQValues(state)
        # q_value_arr[step] = q_values
        # state_arr[step] = state
        # print(f"Q-values: {q_values}")

        # print("Selecting action...")
        action = dqn.selectAction(q_values, exploration_rate)
        # print(f"Action selected: {action}")

        # print("Taking step in the environment...")
        next_state, reward, done, _ = env.step(action)
        # print(f"Step taken. Next state: {next_state}, Reward: {reward}, Done: {done}")


        dqn.addMemory(state, action, reward, next_state, done)
        # print(f"Added to memory")

        state = next_state

        total_reward += reward

        # Update the exploration rate
        exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)

        if done:
            print(f"done!")
            break
    
    print(f"about to learn")
    dqn.learnOnMiniBatch(batch_size)
    print(f"learned successfully")


    robot_start_pose = [4.45, -1.7, 0, 0, 0]
    # ensures goal isn't within -0.5 and 0.5 in either direction
    num1 = next(n for n in iter(lambda: random.uniform(-1.5, 1.5), None) if not -0.5 < n < 0.5)
    num2 = next(n for n in iter(lambda: random.uniform(-1.5, 1.5), None) if not -0.5 < n < 0.5)
    goal = [num1 + robot_start_pose[0], num2 + robot_start_pose[1], 0, 0, 0]
    print("new goal:", goal)
    env.set_goal(goal)


    if episode % 3:
        print(f"saving model")
        dqn.saveModel(model_path)

    # Store the total reward for this episode
    reward_list.append(total_reward)

    # Dynamically plot rewards
    plot_rewards_live(reward_list)
    # plot_best_actions(state_arr, q_value_arr)
    # Example usage
    # visualize_policy_2d(dqn, state_space_x, state_space_y)

    # Update the target network periodically
    if episode % 10 == 0:
        dqn.updateTargetNetwork()

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    print(f'state: {state}')
    print(f'q values: {dqn.getQValues(state)}')

# Save the trained model
dqn.saveModel(model_path)
env.close()




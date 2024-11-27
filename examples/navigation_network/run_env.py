import gym
from gym_gazebo.envs.navigation_network.navigation_training_env import NavigationTrainingEnv

# Initialize the environment
env = NavigationTrainingEnv()

# Test the environment
obs = env.reset()
print(f"Initial Observation: {obs}")

for _ in range(1000000):  # Take random actions
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"Action: {action}, Observation: {obs}, Reward: {reward}, Done: {done}")
    if done:
        break

env.close()

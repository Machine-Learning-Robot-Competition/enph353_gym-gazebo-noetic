import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from tianshou.policy import DQNPolicy
from tianshou.data import Collector, ReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import offpolicy_trainer
from gym_gazebo.envs.navigation_network.navigation_training_env import NavigationTrainingEnv
import matplotlib.pyplot as plt

# Hyperparameters
input_size = 4  # Observation size: [x, y, cos(theta), sin(theta)]
output_size = 3  # Number of discrete actions
memory_size = 10000
discount_factor = 0.99
learning_rate = 0.005
batch_size = 64
train_num = 8  # Number of parallel training environments
test_num = 2  # Number of parallel testing environments
epochs = 100
steps_per_epoch = 1000
update_target_freq = 320  # Frequency to update target network
exploration_decay = 0.9991
min_exploration_rate = 0.01
max_episodes = 1000
model_path = "dqn_tianshou_drone_navigation.pth"

# Define the Neural Network for DQN
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, obs, state=None, info=None):
        return self.net(obs), state

# Define environment wrapper
def make_env():
    return NavigationTrainingEnv()

# Create training and testing environments
train_envs = SubprocVectorEnv([make_env for _ in range(train_num)])
test_envs = SubprocVectorEnv([make_env for _ in range(test_num)])

# Create policy and replay buffer
net = Net(input_size, output_size)
optimizer = Adam(net.parameters(), lr=learning_rate)
policy = DQNPolicy(
    model=net,
    optim=optimizer,
    discount_factor=discount_factor,
    estimation_step=1,  # One-step Q-learning
    target_update_freq=update_target_freq,
)
buffer = ReplayBuffer(size=memory_size)

# Collectors for training and testing
train_collector = Collector(policy, train_envs, buffer)
test_collector = Collector(policy, test_envs)

# Dynamic epsilon decay
def linear_decay(epoch, epochs):
    return max(min_exploration_rate, 1.0 - epoch / epochs)

# Training callback
def save_checkpoint(epoch, env_step):
    torch.save(policy.state_dict(), model_path)

# Training loop with Tianshou's offpolicy_trainer
result = offpolicy_trainer(
    policy,
    train_collector,
    test_collector,
    max_epoch=epochs,
    step_per_epoch=steps_per_epoch,
    step_per_collect=10,
    update_per_step=1,
    batch_size=batch_size,
    train_fn=lambda epoch, env_step: policy.set_eps(linear_decay(epoch, epochs)),
    test_fn=lambda epoch, env_step: policy.set_eps(0.05),
    stop_fn=lambda mean_rewards: mean_rewards >= 300,  # Example stop condition
    save_fn=save_checkpoint,
)

# Plot training rewards
plt.plot(result['env_step'], result['rew'])
plt.xlabel('Environment Steps')
plt.ylabel('Rewards')
plt.title('Training Rewards')
plt.show()

# Save the trained model
torch.save(policy.state_dict(), model_path)

# Close environments
train_envs.close()
test_envs.close()

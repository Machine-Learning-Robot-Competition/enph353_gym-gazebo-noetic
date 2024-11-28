#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import gym
import numpy as np

rewards_key = 'episode_rewards'

class LivePlot(object):
    def __init__(self, outdir, data_key=rewards_key, line_color='blue'):
        """
        Liveplot renders a graph of either episode_rewards or episode_lengths
        Args:
            outdir (outdir): Monitor output file location used to populate the graph
            data_key (Optional[str]): The key in the json to graph (episode_rewards or episode_lengths).
            line_color (Optional[dict]): Color of the plot.
        """
        self.outdir = outdir
        self.data_key = data_key
        self.line_color = line_color

        #styling options
        matplotlib.rcParams['toolbar'] = 'None'
        plt.style.use('ggplot')
        plt.xlabel("Episodes")
        plt.ylabel(data_key)
        fig = plt.gcf().canvas.set_window_title('simulation_graph')

    def plot(self, env):
        if self.data_key is rewards_key:
            data = gym.wrappers.Monitor.get_episode_rewards(env)
        else:
            data = gym.wrappers.Monitor.get_episode_lengths(env)

        plt.plot(data, color=self.line_color)

        # pause so matplotlib will display
        # may want to figure out matplotlib animation or use a different library in the future
        plt.pause(0.000001)



def plot_rewards_live(reward_list):
    """
    Dynamically plots total rewards per episode during training.

    Args:
        reward_list (list of float): List of total rewards for each episode.
    """
    plt.ion()  # Turn on interactive mode
    plt.clf()  # Clear the current figure
    plt.figure(1)  # Ensure the correct figure is updated
    plt.plot(reward_list, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Performance: Total Reward per Episode')
    plt.legend()
    plt.grid()
    plt.pause(0.001)  # Pause briefly to update the plot





import matplotlib.pyplot as plt

def visualize_q_values_2d(dqn, state_space_x, state_space_y, action_index):
    """
    Visualizes the Q-values for a specific action in a 2D state space as a heatmap.

    Args:
        dqn: The DeepQ object containing the trained model.
        state_space_x: A list or range of values for the first state dimension.
        state_space_y: A list or range of values for the second state dimension.
        action_index: The action index to visualize Q-values for.
    """
    q_values_grid = np.zeros((len(state_space_x), len(state_space_y)))

    # Generate Q-values for the grid
    for i, x in enumerate(state_space_x):
        for j, y in enumerate(state_space_y):
            state = np.array([x, y, 0.0])  # Assuming 3D state with z = 0.0
            q_values = dqn.getQValues(state)
            q_values_grid[i, j] = q_values[action_index]

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(q_values_grid, extent=(state_space_x[0], state_space_x[-1], state_space_y[0], state_space_y[-1]),
               origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label="Q-value")
    plt.title(f"Q-value Heatmap for Action {action_index}")
    plt.xlabel("State Dimension X")
    plt.ylabel("State Dimension Y")
    plt.show()


def plot_best_actions(states, q_values_array):
    """
    Plots the best actions as vectors in 2D space based on Q-values for multiple states.

    Args:
        states (list of tuples or np.ndarray): Array of 2D states [(x1, y1), (x2, y2), ...].
        q_values_array (list of lists or np.ndarray): Q-values for each state, 
                                                      where each row corresponds to a state's Q-values.
    """
    # Define action directions: (dx, dy)
    action_vectors = {
        0: (1, 0),  # Right
        1: (-1, 0), # Left
        2: (0, 1),  # Up
        3: (0, -1)  # Down
    }

    # Initialize the plot
    plt.figure(figsize=(8, 8))
    plt.grid()
    plt.title("Best Actions in 2D Space")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Plot each state and its best action
    for state, q_values in zip(states, q_values_array):
        best_action = np.argmax(q_values)  # Determine the best action
        direction = action_vectors[best_action]  # Get the corresponding vector

        # Plot the action vector
        plt.quiver(
            state[0], state[1],  # Start point (x, y)
            direction[0], direction[1],  # Vector components (dx, dy)
            angles='xy', scale_units='xy', scale=1, color='blue', linewidth=1
        )
        plt.text(
            state[0], state[1], f"{best_action}", fontsize=8, color='red'
        )  # Optional: Label the best action

    # Set axis limits (adjust as needed)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    # Show the plot
    plt.show()



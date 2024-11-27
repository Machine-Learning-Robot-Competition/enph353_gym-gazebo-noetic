import gym
import rospy
import subprocess
from gym import spaces
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty
import numpy as np
from robot_controller.srv import GoForward
import pathlib
import toml
from geometry_msgs.msg import Twist
from tf.transformations import quaternion_from_euler

CONFIG_PATH: str = str(pathlib.Path(__file__).absolute().parent.parent.parent.parent.parent / 'robot_controller' / "config" / "robot.toml")
with open(CONFIG_PATH) as f:
    robot_config = toml.load(f)

robot_model_name: str = robot_config["info"]["model_name"]
initial_conditions: dict = robot_config["initial_conditions"]


class NavigationTrainingEnv(gym.Env):
    def __init__(self):
        # Initialize the ROS node
        # rospy.init_node('drone_sim_env', anonymous=True)
        # rospy.loginfo("ROS node initialized successfully!")

        # Define the goal position
        self.goal_position = np.array([5.0, 0.0, 5.0])  # Example 3D goal position

        # Launch the simulation
        self.sim_process = subprocess.Popen(
            ['./run_sim.sh'],  # Adjust if needed
            cwd='/home/fizzer/ros_ws/src/2024_competition/enph353/enph353_utils/scripts',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        rospy.loginfo("Simulation launched successfully!")

        rospy.init_node('drone_sim_env', anonymous=True)
        rospy.loginfo("ROS node initialized successfully!")

        # Publisher and subscriber
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/localization_pose', PoseWithCovarianceStamped, self.localization_callback)
        rospy.loginfo("Publisher and subscriber initialized!")

        # Reset service
        rospy.wait_for_service('/gazebo/reset_simulation')
        self.reset_sim_service = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        rospy.wait_for_service('reset_model_service')
        self.reset_model_service = rospy.ServiceProxy('reset_model_service', GoForward)

        # Initialize observation variables
        self.current_pose = None

        # Define the action and observation spaces
        self.action_space = spaces.Discrete(8)  # 8 discrete actions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)  # [x, y, z]

    def localization_callback(self, msg):
        """Callback for /localization_pose topic."""
        pose = msg.pose.pose
        self.current_pose = np.array([pose.position.x, pose.position.y, pose.position.z])  # [x, y, z]
    
    def reset(self):
        """Reset the simulation and return the initial observation."""
        rospy.logwarn("RESET CALLED!")
        self.reset_model_service()  # Reset the simulation
        # try:
        #     rospy.sleep(0.5)  # Allow time for reset
        # except rospy.exceptions.ROSTimeMovedBackwardsException:
        #     rospy.logwarn("Time moved backwards. Retrying sleep...")
        #     rospy.sleep(0.5)  # Retry sleep after the exception

        # # Wait for the first pose to be received
        # timeout = rospy.Time.now() + rospy.Duration(5)
        # while self.current_pose is None and rospy.Time.now() < timeout:
        #     rospy.sleep(0.1)

        # if self.current_pose is None:
        #     rospy.logwarn("Failed to receive pose update after reset!")
        #     return np.zeros(3)  # Return a default observation

        msg = ModelState()
        msg.model_name = robot_model_name

        orientation_quaternion = quaternion_from_euler(initial_conditions["R"], initial_conditions["P"], initial_conditions["Y"])

        msg.pose.position.x = initial_conditions["x"]
        msg.pose.position.y = initial_conditions["y"]
        msg.pose.position.z = initial_conditions["z"]
        msg.pose.orientation.x = orientation_quaternion[0]
        msg.pose.orientation.y = orientation_quaternion[1]
        msg.pose.orientation.z = orientation_quaternion[2]
        msg.pose.orientation.w = orientation_quaternion[3]
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0

        # Extract position as an array of 3 values [x, y, z]
        initial_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        return initial_position



    def step(self, action):
        """Take a step in the environment."""
        # Define discrete actions as [vx, vy, vz, yaw_rate]
        speed = 5
        actions = [
            [speed, 0.0, 0.0, 0.0],   # Move forward
            [-speed, 0.0, 0.0, 0.0],  # Move backward
            [0.0, speed, 0.0, 0.0],   # Move right
            [0.0, -speed, 0.0, 0.0],  # Move left
            [0.0, 0.0, speed, 0.0],   # Ascend
            [0.0, 0.0, -speed, 0.0],  # Descend
            [0.0, 0.0, 0.0, speed],   # Rotate clockwise
            [0.0, 0.0, 0.0, -speed],  # Rotate counterclockwise
        ]

        # Get the selected action
        selected_action = actions[action]
        rospy.loginfo(f"Taking step with action: {selected_action}")

        # Publish the action as a velocity command
        vel_cmd = Twist()
        vel_cmd.linear.x = selected_action[0]
        vel_cmd.linear.y = selected_action[1]
        vel_cmd.linear.z = selected_action[2]
        vel_cmd.angular.z = selected_action[3]
        self.cmd_vel_pub.publish(vel_cmd)

        # Wait for the environment to update
        rospy.sleep(0.1)

        # Get the current observation
        obs = self.current_pose if self.current_pose is not None else np.zeros(3)

        # Calculate the Euclidean distance to the goal
        distance_to_goal = np.linalg.norm(obs - self.goal_position)

        # Reward function
        reward = -distance_to_goal  # Penalize distance to the goal

        # Bonus for reaching the goal
        if distance_to_goal < 0.5:  # Goal threshold
            reward += 100  # Bonus reward
            done = True
            rospy.loginfo("Goal reached!")

        # Penalty for going out of bounds
        elif any(abs(coord) > 10 for coord in obs):
            reward -= 50  # Penalty for leaving bounds
            done = True
            rospy.logwarn("Out of bounds!")

        else:
            done = False

        return obs, reward, done, {}

    def close(self):
        """Shutdown the simulation."""
        if self.sim_process:
            self.sim_process.terminate()
            try:
                self.sim_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.sim_process.kill()
                rospy.logwarn("Simulation process forcefully terminated.")
        rospy.signal_shutdown("Environment closed.")

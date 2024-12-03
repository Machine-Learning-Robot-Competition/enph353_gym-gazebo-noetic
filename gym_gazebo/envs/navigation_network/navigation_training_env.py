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
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import Vector3Stamped, Point, Quaternion

import threading

CONFIG_PATH: str = str(pathlib.Path(__file__).absolute().parent.parent.parent.parent.parent / 'robot_controller' / "config" / "robot.toml")
DATABASE_PATH = "/path/to/saved/database/rtabmap.db"  # Replace with the actual path

input_size = 5
output_size = 3

with open(CONFIG_PATH) as f:
    robot_config = toml.load(f)

robot_model_name: str = robot_config["info"]["model_name"]
initial_conditions: dict = robot_config["initial_conditions"]


class NavigationTrainingEnv(gym.Env):
    def __init__(self):
        self.goal_position = np.array([2.0, 0.5, 0.0, 0.0, 0.0])

        # Launch the simulation
        self.sim_process = subprocess.Popen(
            ['./run_sim.sh'],
            cwd='/home/fizzer/ros_ws/src/2024_competition/enph353/enph353_utils/scripts',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        rospy.loginfo("Simulation launched successfully!")

        rospy.init_node('drone_sim_env', anonymous=True)
        rospy.loginfo("ROS node initialized successfully!")

        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/localization_pose', PoseWithCovarianceStamped, self.localization_callback)
        rospy.loginfo("Publisher and subscriber initialized!")

        rospy.wait_for_service('reset_model_service')
        self.reset_model_service = rospy.ServiceProxy('reset_model_service', GoForward)

        # Subscribe to the fix_velocity topic from hector_quadrotor (geometry_msgs/Vector3Stamped)
        self.velocity_sub = rospy.Subscriber('/fix_velocity', Vector3Stamped, self.velocity_callback)

        # Initialize observation variables
        self.relative_pose = None
        self.current_pose = None
        self.last_pose = None


        # Define the bounds for the observation space
        low = np.array([-np.inf, -np.inf, -np.inf, -1, -1], dtype=np.float32)  # x, y, z: unbounded; theta: 0
        high = np.array([np.inf, np.inf, np.inf, 1, 1], dtype=np.float32)  # x, y, z: unbounded; theta: 2pi

        # Define the observation space
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(output_size)

        self.current_cmd = Twist() 
        self._stop_event = threading.Event()
        self._publisher_thread = threading.Thread(target=self._publish_cmd_vel)
        self._publisher_thread.start()

        self.velocity_vector = [0, 0, 0]
        self.distance_vector = [0, 0, 0]
        self.unmoving_count = 0
    
    def velocity_callback(self, msg):
        # Extract linear velocity from Vector3Stamped message
        vx = msg.vector.x
        vy = msg.vector.y
        vz = msg.vector.z  #
        self.velocity_vector = [vx, vy]

    def _publish_cmd_vel(self):
        """Continuously publish the current velocity command to /cmd_vel."""
        rate = rospy.Rate(45) # publishing rate in Hz
        while not self._stop_event.is_set() and not rospy.is_shutdown():
            # self.cmd_vel_pub.publish(self.current_cmd)
            rate.sleep()

    def set_goal(self, coords):
        self.goal_position = coords


    def localization_callback(self, msg):
        """Callback for /localization_pose topic."""

        pose = msg.pose.pose
        orientation = pose.orientation
        theta = self.get_theta_from_orientation(orientation)
        self.current_pose = np.array([pose.position.x, pose.position.y, pose.position.z, -np.cos(theta), -np.sin(theta)])
        self.relative_pose = self.goal_position - self.current_pose
    
    def reset(self):
        """Reset the simulation and return the initial observation."""
        rospy.logwarn("RESET CALLED!")
        self.reset_model_service()  # Reset the simulation

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

        quaternion = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        )
        _, _, yaw = euler_from_quaternion(quaternion)
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
            msg.pose.position.z,
            np.cos(yaw),
            np.sin(yaw),
        ])
        
        self.last_pose = initial_position
        self.current_pose = initial_position
        self.relative_pose = self.goal_position - initial_position

        relative_pose_no_z = [self.relative_pose[0], self.relative_pose[1], self.relative_pose[3], self.relative_pose[4]]
        return np.array(relative_pose_no_z)
    

    def step(self, action):
        """Take a step in the environment."""
        done = False
        
        speed = 1.0
        yaw_speed  = speed * 1.2
        actions = [
            [speed, 0.0, 0.0, 0.0],   # Move forward
            # [-speed, 0.0, 0.0, 0.0],  # Move backward
            [0.0, 0.0, 0.0, yaw_speed],   # Rotate clockwise
            [0.0, 0.0, 0.0, -yaw_speed],  # Rotate counterclockwise
        ]

        selected_action = actions[action]
        rospy.loginfo(f"Taking step with action: {selected_action}")
        self.current_cmd.linear.x = selected_action[0]
        self.current_cmd.linear.y = selected_action[1]
        self.current_cmd.linear.z = selected_action[2]
        self.current_cmd.angular.z = selected_action[3]

        rospy.sleep(0.7)

        # Get the current observation
        obs = self.relative_pose if self.relative_pose is not None else np.zeros(5)

        # Calculate the Euclidean distance to the goal
        self.distance_vector = self.relative_pose[:3]
        distance_to_goal = np.linalg.norm(self.distance_vector)
        distance_traveled = np.linalg.norm(obs[:3] - self.last_pose[:3])
        
        if distance_traveled < 0.05:
            self.unmoving_count += 1
        else:
            self.unmoving_count = 0

        # Reward function
        reward = self.calculate_reward()

        self.last_pose = obs

        if distance_to_goal > 5:  # Penalty for going out of bounds
            reward -= 1
            done = True
            rospy.logwarn("Out of bounds!")
        elif distance_to_goal < 0.05:
            rospy.logwarn("__________________GOAL REACHED!__________________")
            done = True
        elif self.unmoving_count > 7:
            reward -= 1
            rospy.logwarn("STUCK!")
            done = True
        else:
            done = False

        # print(f'current pose: {self.current_pose}')
        # print(f'desired pose: {self.goal_position}')
        print(f'relative pose: {self.relative_pose}')   
        
        obs_no_z = np.array([obs[0], obs[1], obs[3], obs[4]])
        return obs_no_z, reward, done, {}

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

    def calculate_reward(self):
        velocity = self.velocity_vector[:2]
        velocity_vector_norm = velocity / np.linalg.norm(velocity)
        distance_vector = self.distance_vector[:2]
        distance_norm = np.linalg.norm(distance_vector)

        if distance_norm < 0.05:
            return 10.0 # the jackpot!

        # Normalize the distance vector
        distance_vector_norm = distance_vector / distance_norm

        dot_product = np.dot(velocity_vector_norm, distance_vector_norm)

        # Gradient-based reward
        alignment_reward = dot_product  # Reward alignment with goal direction
        distance_weight = 1 / (1 + distance_norm)  # Closer gets a higher weight
        direction_weight = 1
        if np.abs(dot_product) > 0.8:
            direction_weight = 1.5

        # Final reward scales with alignment and distance weighting
        reward = alignment_reward * distance_weight * direction_weight

        reward_norm = np.tanh(reward)

        print(f'_____________PRESCALED || {reward}')
        print(f'_____________  SCALED  || {reward_norm}')
        print(f'_____________  DISTANCE  || {distance_norm}')

        return reward_norm
    
    def get_theta_from_orientation(self, orientation):
        """
        Extract the yaw angle (theta) from a quaternion.

        Args:
            orientation: A geometry_msgs/Quaternion message containing x, y, z, w.

        Returns:
            theta: The yaw angle in radians.
        """
        quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
        _, _, yaw = euler_from_quaternion(quaternion)
        return yaw
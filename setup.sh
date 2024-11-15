#!/bin/bash

cd $(realpath ~)/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src
echo "Linking $(realpath ~)/enph353_ws/src repositories to $(realpath ~)/enph353_gym-gazebo-noetic/gym_gazebo/envs/ros_ws/src..."

ln -s ~/enph353_ws/src/2024_competition
echo "Linked 2024_competition"

ln -s ~/enph353_ws/src/hector_quadrotor_noetic
echo "Linked hector_quadrotor_noetic"

ln -s ~/enph353_ws/src/hector_quadrotor_noetic
echo "Linked hector_quadrotor_noetic"

ln -s ~/enph353_ws/src/robot_controller
echo "Linked robot_controller"

echo "Ensure to catkin_make!"

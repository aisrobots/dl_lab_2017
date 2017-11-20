# Simple Visual Planner Exercise
This folder contains the coded that you need to get started with the visual planner learning exercise.

# Overview
There are five main files that should be of interest for you:

  The file astar_demo.py contains a demo script that displays how an optimal A* planner would solve the planning problem this is roughly the performance that your agent can achieve in the end. You will train your agent to imitate this A* planner. Keep in mind though that the A* algorithm has full knowledge about the world and how transitions happen whereas your agent only gets a local view of the environment via an image.

  The get_data.py script generates some training data for you and stores by running the A* planner for a large number of steps, collecting the corresponding images and executed optimal actions, and saves the resulting data into a file.

  The train_agent.py script is the file you should adapt to train a neural network to predict the actions of the A* planner.
  
  The test_agent.py script is meant to be run after you have trained your agent and you should adapt it to test how well your agent is doing.

  The utils.py file contains some additional options that you might find useful NOTE: especially the disp_on variable might be interesting as it switches between running the scripts with and without the visualization on

Additionally you can change the map layour (or add new maps) by drawing a map in the format defined in maps.py.
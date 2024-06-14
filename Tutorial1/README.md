# Tutorial 1 - Self Driving Car

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)


## Description

Welcome to the self-driving car simulation tutorial! In this tutorial, you will learn how to create a self-driving car experience using reinforcement learning techniques. Here, we will guide you through the steps to generate a 2D map, calculate the shortest route between two points, and train a car to navigate this route.

### Overview
This tutorial will cover:

* Generating a 2D Map: We will use a graph to generate a 2D map. The map will include various graphical elements such as roads, markings, houses, and trees.
* Calculating the Shortest Route: Using graph algorithms, we will calculate the shortest route between two points on the map.
* Reinforcement Learning: We will train a virtual car to drive between the two points using reinforcement learning techniques.

### Prerequisites
To follow along with this tutorial, you should have a basic understanding of:

* Python programming
* Graph theory and algorithms
* Reinforcement learning concepts

### Tools and Libraries
We will be using the following tools and libraries:

* raylib: A simple and easy-to-use library to render the 2D map and graphical elements.
* Python: For implementing the algorithms and reinforcement learning logic.

## Getting Started

### Dependencies
* Python 3.11 or above.
* Poetry 1.6.1 or above.
* Just 1.28.0 or above.

### Install and setup locally
Run the following command line:

```bash
just install
```

### Run the tutorial
To run the tutorial1:

```bash
just run
```

### Display the coverage report
Run the following command line:

```bash
just coverage
```

### Documentation

#### Generating a 2D Map
We will create a function to generate a 2D map. The map will be randomly generated but will always produce the same map for a given seed, ensuring reproducibility. The map will include various graphical elements such as:

* Roads
* Markings
* Houses
* Trees

#### Calculating the Shortest Route
Using graph theory, we will calculate the shortest route between two points on the map. This involves:

* Representing the map as a graph
* Implementing algorithms such as Dijkstra's or A* to find the shortest path

#### Implementing Reinforcement Learning
We will implement a reinforcement learning algorithm to train the car to navigate from the starting point to the destination. This involves:

* Defining the state and action space
* Setting up the reward system
* Training the car using a reinforcement learning algorithm such as Q-learning or Deep Q-Networks (DQN)

#### Rendering the Simulation
Using raylib, we will render the map and visualize the car's journey as it learns to drive between the two points.

#### Conclusion
By the end of this tutorial, you will have a working self-driving car simulation. You will understand how to generate a reproducible 2D map, calculate the shortest route, and train a car using reinforcement learning.

Feel free to experiment with different map seeds, reinforcement learning parameters, and algorithms to see how they affect the car's performance.

Happy coding and enjoy your self-driving car simulation experience!

## Contribute

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Authors

* Romuald Rousseau, romuald.rousseau@servier.com

## Version History

* 0.1.0
* Initial Release


# Tutorial 1 - Self Driving Car

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)


## Description

This tutorial is a self driving car experience using reinforcement learning technics. We will first gernerate a 2D map
using a graph, calculate the shortest route between 2 points and let the car learn to drive between these 2 points.

We will use the [raylib](https://www.raylib.com/) library for the rendering of the experience. The 2D map will have graphical
elements such as roads, markings, houses, trees ... The map will be randomly generated but will always give the same map
for a given seed, making the experience reproductible.

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

TBD

## Contribute

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Authors

* Romuald Rousseau, romuald.rousseau@servier.com

## Version History

* 0.1.0
* Initial Release


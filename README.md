# TrafficLightSystemRL-DLW2022-VAlly
This repository contains codes for AATLaS, the Auto-Adaptive Traffic Light System to optimize traffic flow from every junction of Singapore. This is a project for the DLW Hackaton 2022 hosted by MLDA@EEE.

## Problem statement
<p align="center"> <img src="controlling/Sample_Grid.png"/> </p>
Consider the problem: we have a busy crossroad with 4 traffic light clusters. We have traffic camera(s) that can give us information about the direction each car is travelling. What is the optimal green-light time for each travelling direction at the crossroad?

## Approach
TLS utilizes the data from the camera(s) to performs 2 steps:

1. Using the camera data, infer the traffic flow for each of the 12 possible travelling directions at the crossroad.
2. From the traffic flow, calculate the optimal green-light time for one of the two sides of the crossroad.

The first step is performed with a deep neural network model for object detection and tracking. The second step is performed with a deep reinforcement learning model to calculate the green-light time.
* Devpost submission: 
* Video:

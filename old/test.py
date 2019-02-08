#!/usr/bin/env python3

"""
Train a PPO to travel from A to B to C.
Uses the new refactored PPO training, for code readability and 
generally faster prototyping.
"""

import numpy as np
from pprint import pprint
from time import time

from ppo import PPO, train_multithread
from utils import display_graph, visualize_sim
from crazyenv.env import ABCEnv


algo = PPO.load("2")

visualize_sim(algo, ABCEnv, 128, lambda env: env.reset())

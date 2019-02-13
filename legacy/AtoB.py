#!/usr/bin/env python3

"""
Train a PPO to hover a drone at a height of 1m above the origin.
Uses the new refactored PPO training, for code readability and 
generally faster prototyping.
"""

import numpy as np
from pprint import pprint
from time import time

from ppo import PPO, train_multithread
from utils import display_graph, visualize_sim
from crazyenv.env import AtoBEnv

training_vars = {
    'EP_MAX': 10000,
    'EP_LEN': 256,
    'GAMMA': 0.98,
    'A_LR': 1e-4,
    'C_LR': 2e-4,
    'MIN_BATCH_SIZE': 128,
    'UPDATE_STEP': 10,
    'EPSILON': 0.001,
    'ENV': AtoBEnv,
}

algo = PPO(training_vars)

start_t = time()
avg_eps_r = train_multithread(algo, training_vars, num_workers=4)

pprint(training_vars)
print('Time Taken: {:.3f}s'.format(time()-start_t))

display_graph(avg_eps_r, N=5)

visualize_sim(algo, training_vars['ENV'], training_vars['EP_LEN'],
    lambda env: env.reset())


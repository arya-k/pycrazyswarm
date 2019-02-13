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
from crazyenv.env2 import StaticObstEnv

training_vars = {
    'EP_MAX': 1000,
    'EP_LEN': 512,
    'GAMMA': 0.98,
    'A_LR': 4e-5,
    'C_LR': 2e-5,
    'MIN_BATCH_SIZE': 128,
    'UPDATE_STEP': 10,
    'EPSILON': 0.001,
    'ENV': StaticObstEnv,
}

algo = PPO(training_vars)

start_t = time()
avg_eps_r = train_multithread(algo, training_vars, num_workers=1)

algo.save('obstacle1')

pprint(training_vars)
print('Time Taken: {:.3f}s'.format(time()-start_t))

display_graph(avg_eps_r, N=5)

visualize_sim(algo, training_vars['ENV'], training_vars['EP_LEN'],
    lambda env: env.reset())


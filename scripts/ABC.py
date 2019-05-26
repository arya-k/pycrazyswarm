#!/usr/bin/env python3

"""
Train a PPO to travel from A to B to C.
Uses the new refactored PPO training, for code readability and 
generally faster prototyping.

LEGACY. after this code was written, we switched from a custom PPO script to StableBaselines.
"""

import numpy as np
from pprint import pprint
from time import time

from ppo import PPO, train_multithread
from utils import display_graph, visualize_sim
from crazyenv.env import ABCEnv

# TODO: Serializable PPO Object

training_vars = {
    'EP_MAX': 3000,
    'EP_LEN': 1024,
    'GAMMA': 0.98,
    'A_LR': 1e-4,
    'C_LR': 2e-4,
    'MIN_BATCH_SIZE': 256,
    'UPDATE_STEP': 10,
    'EPSILON': 0.0005,
    'ENV': ABCEnv,
}

algo = PPO(training_vars)

start_t = time()
avg_eps_r = train_multithread(algo, training_vars, num_workers=8)

pprint(training_vars)
print('Time Taken: {:.3f}s'.format(time()-start_t))

algo.save('4')

display_graph(avg_eps_r, N=20)

visualize_sim(algo, training_vars['ENV'], training_vars['EP_LEN'],
              lambda env: env.reset())

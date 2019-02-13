#!/usr/bin/env python3

"""
This utils.py file contains some utility functions that
allow us to more easily visualize the data and the trained
models.
"""

import numpy as np
import matplotlib.pyplot as plt

def display_graph(avg_eps_r, N=1):
    ''' Visualize the average episode reward over time, with smoothing. '''
    smooth_r = np.convolve(avg_eps_r, np.ones((N,))/N, mode='valid')[:-N] # moving avg
    plt.plot(np.arange(len(smooth_r)), smooth_r) # plot the points
    plt.xlabel('Episode')
    plt.ylabel('Average Moving Reward')
    plt.show()


def visualize_sim(algo, env, nsteps, reset_func):
    ''' Visualize the trained model using the env's simulator. '''
    vis_env = env() # create an env
    while True:
        s = reset_func(vis_env) # reset according to the function
        r_tot = 0. # total reward
        for t in range(nsteps): # nsteps is the episode length
            a = algo.det_action(s) # get an action from mu (no sigma)
            s, r, *_ = vis_env.step(a)
            vis_env.render()
            r_tot += r
        print('Total reward: {:.3f}'.format(r_tot))

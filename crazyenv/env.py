#!/usr/bin/env python3

import gym
import numpy as np
from .sim import Simulator

CFG = [{'id':1, 'pos':[0., 0., 0.]}]

class HoverEnv(gym.Env):
    """ Gym Env to train a single drone to hover at 1m. """
    def __init__(self):
        self.max_speed = 0.1 #m/s in a given direction
        self.dt = 0.1
        self.action_dim = 3
        self.observation_dim = 3

        self.renderer = None
        self.sim = Simulator(self.dt, CFG)

    def step(self, u):
        u = np.clip(u, -self.max_speed, self.max_speed)
        self.sim.crazyflies[0].goTo(u, 0., 1., self.sim.t)

        self.sim.t += self.sim.dt

        obs = self.sim.crazyflies[0].position(self.sim.t)
        rpy = self.sim.crazyflies[0].rpy(self.sim.t)
        diff = np.array([0.,0.,1.]) - obs

        cost = 4. * np.sqrt(diff.dot(diff))
        cost += 0.05 * u.dot(u)
        cost += 0.05 * rpy.dot(rpy)

        return obs, -cost, self.sim.t > 10, {}

    def reset(self):
        self.sim.t = 0.
        self.sim._init_cfs()
        self.sim.crazyflies[0].takeoff(0., 1., -1) # prep for goTo commands
        return np.array([0., 0., 0.])

    def render(self):
        if self.renderer is None:
            from .vis.visVispy import VisVispy
            self.renderer = VisVispy()
        self.renderer.update(self.sim.t, self.sim.crazyflies)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = None

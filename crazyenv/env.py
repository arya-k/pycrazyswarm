#!/usr/bin/env python3

import gym
import numpy as np
from .sim import Simulator
from collections import deque

class HoverEnv(gym.Env):
    """ Gym Env to train a single drone to hover at 1m. """

    act_dim = 3
    obs_dim = 3

    def __init__(self):
        self.max_speed = 0.1 #m/s in a given direction
        self.dt = 0.1

        self.renderer = None
        self.sim = Simulator(self.dt)

    def step(self, u):
        u = self.max_speed * np.array(u).astype(np.float64)
        self.sim.crazyflies[0].goTo(u, 0., 1., self.sim.t)

        self.sim.t += self.sim.dt

        obs = self.sim.crazyflies[0].position(self.sim.t)
        reward = self._reward(obs, u) 
        return obs, reward, self.sim.t > 10, {}

    def _reward(self, obs, u):
        diff = np.array([0., 0., 1.]) - obs
        u_mag = np.linalg.norm(u)
        diff_mag = np.linalg.norm(diff)
        u_ = u / u_mag if u_mag else np.zeros(u.shape)
        diff_ = diff / diff_mag if diff_mag else np.zeros(diff.shape)
        cos_theta = np.clip(np.dot(u_, diff_), -1., 1.)

        dir_reward = 1. / (1.1 - cos_theta)
        dist_reward = 1. / (.1 + diff_mag)
        movement_cost = u_mag * 1

        return dir_reward + dist_reward - movement_cost

    def reset(self, loc=[0,0,0]):
        self.sim.t = 0.
        self.sim._init_cfs([{'id':1, 'pos':[loc[0], loc[1], 0]}])
        self.sim.crazyflies[0].takeoff(loc[2], 10., -10)
        return np.array(loc)

    def render(self):
        if self.renderer is None:
            from .vis.visVispy import VisVispy
            self.renderer = VisVispy()
        self.renderer.update(self.sim.t, self.sim.crazyflies)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = None


class AtoBEnv(gym.Env):
    """ Gym Env to train a single drone to travel between two arbitrary points. """

    act_dim = 3
    obs_dim = 6

    def __init__(self):
        self.max_speed = 0.1 #m/s in a given direction
        self.dt = 0.1

        self.renderer = None
        self.sim = Simulator(self.dt)

    def step(self, u):
        u = self.max_speed * np.array(u).astype(np.float64)
        self.sim.crazyflies[0].goTo(u, 0., 1., self.sim.t)

        self.sim.t += self.sim.dt

        obs = self._obs()
        reward = self._reward(obs, u) 
        return obs, reward, False, {}

    def _obs(self):
        pos = self.sim.crazyflies[0].position(self.sim.t)
        return np.append(pos, self.target)

    def _reward(self, obs, u):
        diff = obs[3:] - obs[:3]
        u_mag = np.linalg.norm(u)
        diff_mag = np.linalg.norm(diff)

        dist_reward = 1. / (.1 + diff_mag)
        return dist_reward +  u_mag/(1+dist_reward)

    def reset(self):
        self.sim.t = 0.
        self.target = np.random.normal([0,0,1], [.3,.3,.3])

        loc = np.random.normal(self.target, [.3,.3,.3])
        self.sim._init_cfs([{'id':1, 'pos':[loc[0], loc[1], 0]}])
        self.sim.crazyflies[0].takeoff(loc[2], 10., -10)
        return self._obs()

    def render(self):
        if self.renderer is None:
            from .vis.visVispy import VisVispy
            self.renderer = VisVispy()
        self.renderer.update(self.sim.t, self.sim.crazyflies, [self.target])

    def close(self): 
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = None


class ABCEnv(gym.Env):
    """ Gym Env to train a single drone to travel between three arbitrary points. """

    act_dim = 3
    obs_dim = 10

    def __init__(self):
        self.max_speed = 0.1 #m/s in a given direction
        self.dt = 0.1

        self.renderer = None
        self.sim = Simulator(self.dt)

    def step(self, u):
        u = self.max_speed * np.array(u).astype(np.float64)
        self.sim.crazyflies[0].goTo(u, 0., 1., self.sim.t)

        self.sim.t += self.sim.dt

        obs = self._obs()
        reward = self._reward(u) 
        return obs, reward, False, {}

    def _obs(self):
        pos = self.sim.crazyflies[0].position(self.sim.t)
        return np.append(pos, [*self.B, *self.C, self.maxB])

    def _point_reward(self, p, u):
        diff = p - self.sim.crazyflies[0].position(self.sim.t)
        u_mag = np.linalg.norm(u)
        diff_mag = np.linalg.norm(diff)

        dist_reward = 1. / (.1 + diff_mag)
        return dist_reward +  u_mag/(1+dist_reward)

    def _reward(self, u):
        self.maxB = max(self.maxB, self._point_reward(self.B, u))
        return self._point_reward(self.C, u) + 3 * self.maxB

    def reset(self):
        self.sim.t = 0.
        A = np.random.normal([0,0,1], [.3,.3,.3])
        self.B = np.random.normal([0,0,1], [.3,.3,.3])
        self.C = np.random.normal([0,0,1], [.3,.3,.3])
        
        self.sim._init_cfs([{'id':1, 'pos':[A[0], A[1], 0]}])
        self.sim.crazyflies[0].takeoff(A[2], 10., -10)
        self.maxB = self._point_reward(self.B, np.zeros(3))
        return self._obs()

    def render(self):
        if self.renderer is None:
            from .vis.visVispy import VisVispy
            self.renderer = VisVispy()
        self.renderer.update(self.sim.t, self.sim.crazyflies, [self.B, self.C])

    def close(self): 
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = None

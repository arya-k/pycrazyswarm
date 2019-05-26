#!/usr/bin/env python3

import gym
import numpy as np
from gym.spaces import Box
from .sim import Simulator
from collections import deque


class HoverEnv(gym.Env):
    """ Gym Env to train a single drone to hover at 1m. """

    def __init__(self):
        self.max_speed = 0.1  # m/s in a given direction
        self.dt = 0.1

        high_obs = np.array([np.finfo(np.float64).max]*3)
        self.action_space = Box(
            low=-self.max_speed, high=self.max_speed, shape=(3,), dtype=np.float64)
        self.observation_space = Box(
            low=-high_obs, high=high_obs, dtype=np.float64)

        self.renderer = None
        self.sim = Simulator(self.dt)

    def step(self, u):
        u = self.max_speed * np.array(u).astype(np.float64)
        self.sim.crazyflies[0].goTo(u, 0., 1., self.sim.t)

        self.sim.t += self.sim.dt

        obs = self.sim.crazyflies[0].position(self.sim.t)
        reward = self._reward(obs, u)
        return obs, reward, self.sim.t > 20, {}

    def _reward(self, obs, u):
        diff = np.array([0., 0., 1.]) - obs
        movement_cost = np.linalg.norm(u) * 0.1
        return np.dot(u, diff) - movement_cost

    def reset(self, loc=[0, 0, 0]):
        self.sim.t = 0.
        self.sim._init_cfs([{'id': 1, 'pos': [loc[0], loc[1], 0]}])
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
        self.max_speed = 0.1  # m/s in a given direction
        self.dt = 0.1

        self.renderer = None
        self.sim = Simulator(self.dt)

    def step(self, u):
        u = self.max_speed * np.array(u).astype(np.float64)
        self.sim.crazyflies[0].goTo(u, 0., 1., self.sim.t)

        self.sim.t += self.sim.dt

        obs = self._obs()
        reward = self._reward(obs, u)
        return obs, reward, np.linalg.norm(obs[:3]-obs[3:]) < 1e-2, {}

    def _obs(self):
        pos = self.sim.crazyflies[0].position(self.sim.t)
        return np.append(pos, self.target)

    def _reward(self, obs, u):
        diff = obs[3:] - obs[:3]
        u_mag = np.linalg.norm(u)
        diff_mag = np.linalg.norm(diff)

        dist_reward = 1. / (.1 + diff_mag)
        return dist_reward + u_mag/(1+dist_reward)

    def reset(self):
        self.sim.t = 0.
        self.target = np.random.normal([0, 0, 0], [.3, .3, .3])
        self.A = np.random.normal(self.target, [.3, .3, .3])
        self.sim._init_cfs([{'id': 1, 'pos': [self.A[0], self.A[1], 0]}])
        self.sim.crazyflies[0].takeoff(self.A[2], 10., -10)
        return self._obs()

    def render(self):
        if self.renderer is None:
            from .vis.visVispy import VisVispy
            self.renderer = VisVispy()
        self.renderer.update(self.sim.t, self.sim.crazyflies, [
                             (self.A, '#0000FF'), (self.target, '#FF0000')])

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = None


class ABCEnv(gym.Env):
    """ Gym Env to train a single drone to travel between three arbitrary points. """

    act_dim = 3
    obs_dim = 12

    def __init__(self):
        self.max_speed = 0.1  # m/s in a given direction
        self.dt = 0.1

        self.renderer = None
        self.sim = Simulator(self.dt)

    def step(self, u):
        u = self.max_speed * np.array(u).astype(np.float64)
        self.sim.crazyflies[0].goTo(u, 0., 1., self.sim.t)

        self.sim.t += self.sim.dt

        obs = self._obs()
        reward = self._reward(u, obs)
        return obs, reward, False, {}

    def _obs(self):
        pos = self.sim.crazyflies[0].position(self.sim.t)
        return np.append(pos, [*self.A, *self.B, *self.C])

    def _reward(self, u, obs):
        # relative vector of drone to circle center
        PO = obs[:3] - self.circle[0]
        # normal vector to circle
        n = np.cross(self.B - self.A, self.C - self.A)

        PO_norm = (np.dot(PO, n) / np.dot(n, n)) * n  # normal component of PO
        PO_r = PO - PO_norm  # relative to center projection of PO onto plane
        # closest point
        p_cl = PO_r/np.linalg.norm(PO_r)*self.circle[1] + self.circle[0]

        rel_p_mid = (p_cl + self.C - 2*self.circle[0])
        # midpoint relative to center
        rel_p_mid *= self.circle[1] / np.linalg.norm(rel_p_mid)

        # correct direction
        c1 = np.cross(self.A - self.circle[0], self.B - self.circle[0])
        c2 = np.cross(p_cl - self.circle[0], rel_p_mid)  # rewarded direction

        if np.dot(c1, c2) < 0:  # if rewarding wrong direction,
            rel_p_mid *= -1  # swtich

        p_mid = rel_p_mid + self.circle[0]  # absolute midpoint

        new_circ = self._fit_circle(
            obs[:3], p_mid, self.C)  # new fitted circle

        o_proj = self.circle[0] + PO_r  # absolute projection drone onto circle
        u_des = np.cross(o_proj - new_circ[0], n)  # vector perpendicular to
        if np.linalg.norm(p_mid - (o_proj + u_des)) > np.linalg.norm(p_mid - (o_proj - u_des)):
            u_des *= -1

        cos_theta = np.dot(u, u_des)/np.linalg.norm(u)/np.linalg.norm(u_des)
        closest_distance = np.linalg.norm(obs[:3]-p_cl)
        return cos_theta * np.linalg.norm(u)

    def _fit_circle(self, A, B, C):
        a = np.linalg.norm(A-B)
        b = np.linalg.norm(B-C)
        c = np.linalg.norm(C-A)

        temp = np.linalg.norm(np.cross(A-B, B-C))

        R = a*b*c/(2 * temp)
        alpha = b**2 * np.dot(A-B, A-C)/(2 * temp**2)
        beta = c**2 * np.dot(B-A, B-C)/(2 * temp**2)
        gamma = a**2 * np.dot(C-A, C-B)/(2 * temp**2)
        P = alpha*A + beta*B + gamma*C

        return P, R

    def reset(self):
        self.sim.t = 0.
        A = np.random.normal([0, 0, 1], [.3, .3, .3])
        B = np.random.normal([0, 0, 1], [.3, .3, .3])
        C = np.random.normal([0, 0, 1], [.3, .3, .3])

        while not np.cross(B-A, C-A).any():  # if colinear
            A = np.random.normal([0, 0, 1], [.3, .3, .3])

        if np.dot(B-A, C-B) < 0:  # B - A - C
            A, B = B, A
        if np.linalg.norm(B-A) > np.linalg.norm(C-A):  # A - C - B
            B, C = C, B

        self.A = A
        self.B = B
        self.C = C
        self.circle = self._fit_circle(self.A, self.B, self.C)

        self.sim._init_cfs([{'id': 1, 'pos': [A[0], A[1], 0]}])
        self.sim.crazyflies[0].takeoff(A[2], 10., -10)

        if self.renderer is not None:
            self.renderer.update(self.sim.t, self.sim.crazyflies)

        return self._obs()

    def render(self):
        if self.renderer is None:
            from .vis.visVispy import VisVispy
            self.renderer = VisVispy()
        self.renderer.update(self.sim.t, self.sim.crazyflies,
                             spheres=[(self.A, '#FF0000'), (self.B,
                                                            '#00FF00'), (self.C, '#0000FF')],
                             obstacles=[((0, 0, 0), .1)],
                             crumb=self.sim.crazyflies[0].position(self.sim.t))

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = None

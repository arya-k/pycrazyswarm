import gym
import numpy as np
from gym.spaces import Box
from multiprocessing import Process, Pool, TimeoutError, Queue
from .sim import Simulator
from itertools import combinations


def split_list(l, sublist_size):
    for i in range(0, len(l), sublist_size):
        yield np.array(l[i:i+sublist_size])


class PFController():
    """ Potential field controller to model obstacles. """

    LAMBDA_1 = 1.  # scale of A -> B field
    LAMBDA_2 = 0.02  # scale of obstacle avoidance field
    P_STAR = 1.  # ignore obstacles more than this far away.

    def error(positions, loc, target=[0., 0., 0.]):
        """ calculate the error at a specified loc location. """
        f_1 = .5 * PFController.LAMBDA_1 * np.dot(target-loc, target-loc)
        f_2 = 0
        for position in positions:
            dist = np.linalg.norm(loc - position)
            dist = max(dist, 1e-5)  # minimum bound to avoid NaNs
            if dist < 1e-5:
                continue
            if (dist < PFController.P_STAR):
                f_2 += .5 * PFController.LAMBDA_2 / (dist*dist)
        return f_2 + f_1

    def gradient(positions, loc, target=[0., 0., 0.]):
        """ calculate the gradient at a specified loc location. """
        v_1 = -1 * PFController.LAMBDA_1 * (loc - target)
        v_2 = np.array([0.]*3)
        for position in positions:
            dist = np.linalg.norm(loc - position)
            if dist < 1e-5:
                continue
            dist = max(dist, 1e-5)  # minimum bound to avoid NaNs
            if (dist < PFController.P_STAR):
                v_2 += (PFController.LAMBDA_2 / (dist**4)) * \
                    (loc - position)
        return v_2 + v_1

    def isColliding(obs):
        """ Calculates whether we should stop the episode because drones collide."""
        for a, b in combinations(obs, 2):
            a, b = np.array(a), np.array(b)
            if (a-b).dot(a-b) < .1**2:
                return True
        return False


class StaticSwarmEnv(gym.Env):

    def __init__(self, num_robots):
        self.max_speed = 0.1  # m/s in a given direction
        self.dt = 0.1
        self.num_robots = num_robots

        high_obs = np.ones((num_robots*10,)) * np.finfo(np.float64).max
        self.action_space = Box(
            low=-self.max_speed, high=self.max_speed, shape=(num_robots*3,), dtype=np.float64)
        self.observation_space = Box(
            low=-high_obs, high=high_obs, dtype=np.float64)
        # self.target = np.array([0., 0., 0.])

        self.renderer = None
        self.sim = Simulator(self.dt)

    def step(self, actions):
        for cf, action in zip(self.sim.crazyflies, split_list(actions, 3)):
            action = self.max_speed * np.array(action).astype(np.float64)
            cf.goTo(action, 0., 1., self.sim.t)
        self.sim.t += self.sim.dt

        obs = self._obs()
        reward = self._reward(obs, actions)
        return obs, reward, self.sim.t > 30, {}

    def _obs(self):
        positions = [cf.position(self.sim.t) for cf in self.sim.crazyflies]
        observations = []
        for pos in positions:
            observations.extend(pos)
            observations.extend([0., 0., 0.])
            observations.append(PFController.error(positions, pos))
            observations.extend(PFController.gradient(positions, pos))
        return observations

    def _reward(self, obs, actions):
        value = 0
        for ob, action in zip(split_list(obs, 10), split_list(actions, 3)):
            value += action.dot(ob[:3]) / \
                np.linalg.norm(action)/np.linalg.norm(ob[:3])
        return value

    def reset(self):
        self.sim.t = 0.
        self.sim._init_cfs([{'id': x, 'pos': np.random.normal(size=3)}
                            for x in range(self.num_robots)])
        for cf in self.sim.crazyflies:
            cf.takeoff(cf.pos[2], 10., -10.)
        return self._obs()

    def render(self):
        if self.renderer is None:
            from .vis.visVispy import VisVispy
            self.renderer = VisVispy()
        self.renderer.update(self.sim.t, self.sim.crazyflies)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = None

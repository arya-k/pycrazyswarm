import gym
import numpy as np
from gym.spaces import Box
from .sim import Simulator

class StaticSwarmEnv(gym.Env):
    """Insert useful comment here"""
    def __init__(self):
        self.max_speed = 0.1
        self.dt = 0.1
        self.num_robots = 4
        self.num_envs = 1

        high_obs = np.array([np.finfo(np.float64).max]*10)
        self.action_space = Box(low=-self.max_speed, high=self.max_speed, shape=(3,), dtype=np.float64)
        self.observation_space = Box(low=-high_obs, high=high_obs, dtype=np.float64)

        self.renderer = None
        self.sim = Simulator(self.dt)

       self.model = None

    def step(self, actions):
        return

    def _obs(self):
        positions = np.array([])
        for cf in self.sim.crazyflies:
            positions.append(cf.position(self.sim.t))
        return np.concatenate((positions))

    def set_model(self, model):
        self.model = model

    def reset(self):
        self.sim.t = 0.
        return

    def render(self):
        return

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = None
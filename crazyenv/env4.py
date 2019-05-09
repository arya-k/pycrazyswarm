import gym
import numpy as np
from gym.spaces import Box
from multiprocessing import Process, Pool, TimeoutError, Queue
from .sim import Simulator
from itertools import combinations


def split_list(l, sublist_size):
    for i in range(0, len(l), sublist_size):
        yield np.array(l[i:i+sublist_size])


class DronePFController():
    """ Potential field controller to model obstacles. """

    LAMBDA_1 = 1.  # scale of A -> B field
    LAMBDA_2 = 0.04  # scale of obstacle avoidance field
    P_STAR = .5  # ignore obstacles more than this far away.

    def error(positions, loc, target=[0., 0., 0.]):
        """ calculate the error at a specified loc location. """
        f_1 = .5 * DronePFController.LAMBDA_1 * np.dot(target-loc, target-loc)
        f_2 = 0
        for position in positions:
            dist = np.linalg.norm(loc - position)
            if dist < 1e-5:
                continue
            dist = max(dist, 1e-5)  # minimum bound to avoid NaNs
            if (dist < DronePFController.P_STAR):
                f_2 += .5 * DronePFController.LAMBDA_2 / (dist*dist)
        return f_2 + f_1

    def gradient(positions, loc, target=[0., 0., 0.]):
        """ calculate the gradient at a specified loc location. """
        v_1 = -1 * DronePFController.LAMBDA_1 * (loc - target)
        v_2 = np.array([0.]*3)
        for position in positions:
            dist = np.linalg.norm(loc - position)
            if dist < 1e-5:
                continue
            dist = max(dist, 1e-5)  # minimum bound to avoid NaNs
            if (dist < DronePFController.P_STAR):
                v_2 += (DronePFController.LAMBDA_2 / (dist**4)) * \
                    (loc - position)
        return v_2 + v_1

    def isColliding(obs):
        """ Calculates whether we should stop the episode because drones collide."""
        for a, b in combinations(obs, 2):
            a, b = np.array(a), np.array(b)
            if (a-b).dot(a-b) < .1**2:
                return True
        return False


class ObstPFController():

    """ Potential field controller to model obstacles. """

    LAMBDA_1 = 1.  # scale of A -> B field
    LAMBDA_2 = 0.04  # scale of obstacle avoidance field
    P_STAR = .5  # ignore obstacles more than this far away.

    def __init__(self, A, B, num_obstacles, obstacle_size_range, dynamic=False):
        midpoint = (B + A) / 2
        location_radius = np.abs((B - A) * .4)

        self.target = B
        self.obstacles = []
        for _ in range(num_obstacles):
            obst = [np.random.normal(midpoint, location_radius),
                    np.random.normal(*obstacle_size_range),
                    np.random.normal([0.]*3, [0.001]*3) if dynamic else np.zeros(3)]
            d1 = np.linalg.norm(B - obst[0]) - np.linalg.norm(obst[1]/2)
            d2 = np.linalg.norm(A - obst[0]) - np.linalg.norm(obst[1]/2)

            while (d1 < ObstPFController.P_STAR or d2 < ObstPFController.P_STAR):
                obst = [np.random.normal(midpoint, location_radius),
                        np.random.normal(*obstacle_size_range),
                        np.random.normal([0.]*3, [0.001]*3) if dynamic else np.zeros(3)]
                d1 = np.linalg.norm(B - obst[0]) - np.linalg.norm(obst[1]/2)
                d2 = np.linalg.norm(A - obst[0]) - np.linalg.norm(obst[1]/2)

            self.obstacles.append(obst)

    def error(self, obs):
        """ calculate the error at a specified obs location. """
        # f_1 = .5 * ObstPFController.LAMBDA_1 * np.dot(self.target-obs, self.target-obs)
        f_2 = 0
        for obst in self.obstacles:
            obstacle_position = np.array(
                [c-r if o < c-r else c+r if o > c+r else o for o, c, r in zip(obs, obst[0], obst[1])])
            dist = np.linalg.norm(obs - obstacle_position)
            dist = max(dist, 1e-5)  # minimum bound to avoid NaNs
            if (dist < ObstPFController.P_STAR):
                f_2 += .5 * ObstPFController.LAMBDA_2 / (dist*dist)
        return f_2  # + f_1

    def gradient(self, obs):
        """ calculate the gradient at a specified obs location. """
        # v_1 = -1 * ObstPFController.LAMBDA_1 * (obs - self.target)
        v_2 = np.array([0.]*3)
        for obst in self.obstacles:
            obstacle_position = np.array(
                [c-r if o < c-r else c+r if o > c+r else o for o, c, r in zip(obs, obst[0], obst[1])])
            dist = np.linalg.norm(obs - obstacle_position)
            dist = max(dist, 1e-5)  # minimum bound to avoid NaNs
            if (dist < ObstPFController.P_STAR):
                v_2 += (ObstPFController.LAMBDA_2 / (dist**4)) * \
                    (obs - obstacle_position)
        return v_2  # + v_1

    def hittingObstacles(self, obs):
        """ Calculates whether we should stop the episode because we are too close to obstacles."""
        for obst in self.obstacles:
            obstacle_position = np.array(
                [c-r if o < c-r else c+r if o > c+r else o for o, c, r in zip(obs, obst[0], obst[1])])
            dist = np.linalg.norm(obs[:3] - obstacle_position)
            if dist < 0.05:  # 5cm from an obstacle
                return True
        return False

    def pfc_step(self):
        """Updates the obstacle positions to simulate movement"""
        for obst in self.obstacles:
            obst[0] += obst[2]


class SingleSwarmEnv(gym.Env):

    def __init__(self, num_robots):
        self.max_speed = 0.6  # m/s in a given direction
        self.dt = 0.1
        self.num_robots = num_robots
        self.B = np.random.rand(3)*2-1

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
        return obs, reward, self.sim.t > 30 or DronePFController.isColliding(split_list(obs, 10)), {}

    def _obs(self):
        positions = [cf.position(self.sim.t) for cf in self.sim.crazyflies]
        observations = []
        for pos in positions:
            observations.extend(pos)
            observations.extend(self.B)
            observations.append(
                DronePFController.error(positions, pos, self.B))
            observations.extend(
                DronePFController.gradient(positions, pos, self.B))
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
        self.B = np.random.rand(3)*2-1
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


class DynamicSwarmEnv(gym.Env):

    def __init__(self, num_robots):
        self.max_speed = 0.7  # m/s in a given direction
        self.dt = 0.1
        self.num_robots = num_robots*2
        self.B1 = np.random.rand(3)*2-1
        self.B2 = -self.B1

        high_obs = np.ones((self.num_robots*10,)) * np.finfo(np.float64).max
        self.action_space = Box(
            low=-self.max_speed, high=self.max_speed, shape=(self.num_robots*3,), dtype=np.float64)
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
        self.pfc.pfc_step()

        obs = self._obs()
        reward = self._reward(obs, actions)
        return obs, reward, self.sim.t > 60 or DronePFController.isColliding(split_list(obs, 10)) or any(self.pfc.hittingObstacles(o) for o in split_list(obs, 10)), {}

    def _obs(self):
        positions = [cf.position(self.sim.t) for cf in self.sim.crazyflies]
        ids = [cf.id for cf in self.sim.crazyflies]

        observations = []
        for pos, swarm_id in zip(positions, ids):
            observations.extend(pos)
            if (swarm_id < self.num_robots/2):
                observations.extend(self.B2)
                observations.append(
                    DronePFController.error(positions, pos, self.B2) + self.pfc.error(pos))
                observations.extend(
                    DronePFController.gradient(positions, pos, self.B2) + self.pfc.gradient(pos))
            else:
                observations.extend(self.B1)
                observations.append(
                    DronePFController.error(positions, pos, self.B1) + self.pfc.error(pos))
                observations.extend(
                    DronePFController.gradient(positions, pos, self.B1) + self.pfc.gradient(pos))
        return observations

    def _reward(self, obs, actions):
        value = 0
        for ob, action in zip(split_list(obs, 10), split_list(actions, 3)):
            value += action.dot(ob[: 3])
            np.linalg.norm(action)/np.linalg.norm(ob[: 3])
        return value

    def reset(self):
        self.sim.t = 0.
        self.B1 = np.random.rand(3)*2-1
        self.B2 = -self.B1
        self.pfc = ObstPFController(
            self.B1, self.B2, 6, [[.15]*3, [.07]*3], True)
        self.sim._init_cfs([{'id': x, 'pos': self.B1+np.random.normal(size=3)/4}
                            for x in range(int(self.num_robots/2))] +
                           [{'id': x, 'pos': self.B2+np.random.normal(size=3)/4}
                            for x in range(int(self.num_robots/2), self.num_robots)])
        for cf in self.sim.crazyflies:
            cf.takeoff(cf.pos[2], 10., -10.)
        return self._obs()

    def render(self):
        if self.renderer is None:
            from .vis.visVispy import VisVispy
            self.renderer = VisVispy()
        self.renderer.update(self.sim.t, self.sim.crazyflies,
                             spheres=[(self.B1, '#FF0000'),
                                      (self.B2, '#0000FF')],
                             obstacles=self.pfc.obstacles,)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = None

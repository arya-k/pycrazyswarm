#!/usr/bin/env python3

import math
import numpy as np
from .cfsim import cffirmware as firm

def arr2vec(a):
    """ Converts python/numpy arrays to firmware 3d vector type """
    a = np.array(a).astype(np.float64)
    return firm.mkvec(a[0], a[1], a[2])

class Crazyflie:
    """ Represents the firmware of a single crazyflie drone. """
    def __init__(self, id, pos):
        self.id = id
        self.pos = np.array(pos)

        self.planner = firm.planner()
        firm.plan_init(self.planner)
        self.planner.lastKnownPosition = arr2vec(pos)

    def goTo(self, goal, yaw, duration, t, relative=True):
        firm.plan_go_to(self.planner, relative, 
            arr2vec(goal), yaw, duration, t)
	
    def takeoff(self, targetHeight, duration, t):
        firm.plan_takeoff(self.planner, self._vposition(t),
            self.yaw(t), targetHeight, duration, t)

    def position(self, t):
        pos = self._vposition(t)
        return np.array([pos.x, pos.y, pos.z])

    def yaw(self, t):
        ev = firm.plan_current_goal(self.planner, t)
        return ev.yaw

    def rpy(self, t):
        ev = firm.plan_current_goal(self.planner, t)
        yaw = ev.yaw
        if self.planner.state == firm.TRAJECTORY_STATE_IDLE:
            acc = np.array([0., 0., 0.])
        else:
            acc = np.array([ev.acc.x, ev.acc.y, ev.acc.z])
        norm = np.linalg.norm(acc)
        if norm < 1e-6:
            return 0., 0., yaw
        else:
            thrust = acc + np.array([0, 0, 9.81])
            z_body = thrust / np.linalg.norm(thrust)
            x_world = np.array([math.cos(yaw), math.sin(yaw), 0])
            y_body = np.cross(z_body, x_world)
            x_body = np.cross(y_body, z_body)
            pitch = math.asin(-x_body[2])
            roll = math.atan2(y_body[2], z_body[2])
            return np.array([roll, pitch, yaw])

    def _vposition(self, t):
        if self.planner.state == firm.TRAJECTORY_STATE_IDLE:
            return self.planner.lastKnownPosition
        else:
            ev = firm.plan_current_goal(self.planner, t)
            self.planner.lastKnownPosition = firm.mkvec(
                ev.pos.x, ev.pos.y, ev.pos.z)
            return firm.mkvec(ev.pos.x, ev.pos.y, ev.pos.z)

class Simulator:
    """ Manages the crazyflies, and keeps track of the clock. """
    def __init__(self, dt, cfg):
        self.t = 0.
        self.dt = dt
        self.cfg = cfg
        self._init_cfs()

    def _init_cfs(self):
        self.crazyflies = [Crazyflie(cf['id'], cf['pos'])
            for cf in self.cfg]



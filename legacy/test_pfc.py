from crazyenv.env2 import StaticObstEnv, PFController
import numpy as np

env = StaticObstEnv()

"""
env.reset()
env.A = np.array([-1,0.1,0]).astype(np.float64)
env.B = np.array([1,0,0]).astype(np.float64)
env.pfc = PFController(env.A, env.B, 1, (np.array([0,0,0]), np.array([0,0,0])))
env.pfc.obstacles = [(np.array([0,0,0]), np.array([.15,.15,.15]))]

env.sim._init_cfs([{'id':1, 'pos':[env.A[0], env.A[1], 0]}])
env.sim.crazyflies[0].takeoff(env.A[2], 10., -10)
"""

while 1:
  env.reset()

  u = np.array([0., 0., 0.]);
  d = False

  while(not d):
    env.render()
    o, e, d, _ = env.step(u);
    u = o[-3:]
  input("Press ENTER to reset env. ")

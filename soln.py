"""
soln.py

Utility script used to compute the optimal
reward for a single episode, using the same
reward function we use for training the HoverEnv
"""

from crazyenv.env import HoverEnv

env = HoverEnv()
total_reward = 0.0

obs = env.reset()
for _ in range(256):
    obs, reward, done, _ = env.step([0,0,1] - obs)
    total_reward += reward

print(total_reward)

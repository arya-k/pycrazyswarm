"""
test.py

A script designed to simulate what is happening
during training, rendering the results and displaying
the reward over episodes to ensure it's operating as
we would expect it to.
"""

from crazyenv.env import HoverEnv
import numpy as np

env = HoverEnv()

NUM_EPS = 100
mean = [0,0,1]
scale = np.array([1,1,.5])

for ep in range(NUM_EPS):
    total_reward = 0.0

    loc = np.random.normal(mean, scale)

    obs = env.reset(loc)
    for _ in range(100):
        obs, reward, done, _ = env.step([0,0,1] - obs)
        env.render()
        total_reward += reward
    print(total_reward)

env.close()

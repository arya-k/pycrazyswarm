import gym
import tensorflow as tf
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

from crazyenv.env import *
from crazyenv.env2 import *
from crazyenv.env3 import *
from crazyenv.env4 import *

env = DummyVecEnv([lambda: HoverEnv()])
model = PPO2.load("Hover1")


# Enjoy trained agent
while True:
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()

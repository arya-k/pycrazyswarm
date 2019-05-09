import gym
import tensorflow as tf
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

from crazyenv.env2 import DynamicObstEnv
from crazyenv.env4 import DynamicSwarmEnv

env = DummyVecEnv([lambda: DynamicSwarmEnv(3)])
model = PPO2.load("Dynamic8")


def split_list(l, sublist_size):
    for i in range(0, len(l), sublist_size):
        yield np.array(l[i:i+sublist_size])


# Enjoy trained agent
while True:
    obs = env.reset()
    done = False
    while not done:
        obs = split_list(obs.reshape((-1,)), 10)
        actions = []
        for ob in obs:
            action, _states = model.predict(ob, deterministic=True)
            actions.extend(list(action))
        actions = np.array(actions)
        obs, rewards, done, info = env.step([actions])
        env.render()

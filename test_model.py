import gym
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

from crazyenv.env2 import StaticObstEnv
env = DummyVecEnv([lambda: StaticObstEnv()])
model = PPO2.load("StaticNoAttraction2")

# Enjoy trained agent
while True:
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        env.render()

"""
First env using multiple drones at once. Initiate N=10
drones at random locations, and then have them converge
towards the origin, without crashing into each other.

Network size had to be increased for this training to work.
"""

import gym
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

from crazyenv.env3 import StaticSwarmEnv

env = DummyVecEnv([lambda: StaticSwarmEnv(10)])

policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[1000, 1000])

model = PPO2(
    MlpPolicy, env,
    gamma=0.98,
    n_steps=256,
    policy_kwargs=policy_kwargs,
    nminibatches=64,
    learning_rate=1e-4,
    cliprange=0.2,
    tensorboard_log='/home/um/tensorboards/',
    verbose=1
)

model.learn(
    total_timesteps=5000000,
    log_interval=1
)

print("Finished training.")

model.save("NaiveSwarm3")
# del model # remove to demonstrate saving and loading
# model = PPO2.load("NaiveSwarm2")

# Enjoy trained agent
while True:
    obs = env.reset()
    done = False
    while not done:
        actions, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(actions)
        env.render()

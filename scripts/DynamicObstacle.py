"""
Train a drone to travel from point A to B, while avoiding obstacles in the way.
Unlike StaticObstEnv, we allow the obstacles to move around in this case.
Obstacles are modeled as particle fields, a. la. PFCs
Training is hyperthreaded (64x), but that means that visualization has
to happen separately (see generate_movie in utils.)
"""

import gym
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

from crazyenv.env2 import DynamicObstEnv

env = SubprocVecEnv([lambda: DynamicObstEnv() for _ in range(64)])

policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[200, 200])

model = PPO2(
    MlpPolicy, env,
    gamma=0.99,
    n_steps=256,
    policy_kwargs=policy_kwargs,
    nminibatches=256,
    learning_rate=1e-4,
    cliprange=0.2,
    tensorboard_log='/home/um/tensorboards/',
    verbose=1
)

model.learn(
    total_timesteps=10000000,
    log_interval=1
)

print("Finished training.")

model.save("Dynamic8")
# del model # remove to demonstrate saving and loading
# model = PPO2.load("2")

# Enjoy trained agent
# while True:
#     obs = env.reset()
#     done = False
#     while not done:
#         action, _states = model.predict(obs, deterministic=True)
#         obs, rewards, done, info = env.step(action)
#         env.render()

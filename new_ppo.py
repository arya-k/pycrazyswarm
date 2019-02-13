import gym
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = DummyVecEnv([lambda: gym.make('CartPole-v1')])

policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[32, 32])

model = PPO2(MlpPolicy, env, verbose=1, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=200000, log_interval=5)

# model.save("ppo2_cartpole")
# del model # remove to demonstrate saving and loading
# model = PPO2.load("ppo2_cartpole")

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

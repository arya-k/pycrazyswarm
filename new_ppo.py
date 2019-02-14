import gym
import tensorflow as tf

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

from crazyenv.env2 import StaticObstEnv
from crazyenv.env import HoverEnv

env = DummyVecEnv([lambda: StaticObstEnv()])

policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[200,200])

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
    total_timesteps=200000,
    log_interval=1
)

model.save("ppo2_staticobst")
# del model # remove to demonstrate saving and loading
# model = PPO2.load("ppo2_cartpole")

# Enjoy trained agent
while True:
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, done, info = env.step(action)
        env.render()


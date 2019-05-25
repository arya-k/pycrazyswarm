import os
import gym
import tensorflow as tf
import numpy as np
import vispy.io as io

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

from crazyenv.env import *
from crazyenv.env2 import *
from crazyenv.env3 import *
from crazyenv.env4 import *


env = DummyVecEnv([lambda: DynamicObstEnv()])
model = PPO2.load("Dynamic8")


# Create a folder
os.system("rm -rf imgs; rm -Rf video.mp4; mkdir imgs")
counter = 0

# Run some episodes
for num_episode in range(10):
    obs = env.reset()
    print(num_episode)

    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()

        # also save the image to an array
        io.write_png("imgs/{0:05}.png".format(counter),
                     env.envs[0].renderer.canvas.render())
        counter += 1

print("Creating video:")
os.system("ffmpeg -r 60 -i imgs/%05d.png -vcodec libx264 -y -an video.mp4 -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -pix_fmt yuv420p && rm -rf imgs")

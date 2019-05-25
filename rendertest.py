import os
import gym
import tensorflow as tf
import numpy as np
import vispy.io as io

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

from crazyenv.env3 import *
from crazyenv.env4 import *

env = DummyVecEnv([lambda: StaticSwarmEnv(10)])
# model = PPO2.load("Dynamic8")


def split_list(l, sublist_size):
    for i in range(0, len(l), sublist_size):
        yield np.array(l[i:i+sublist_size])


# Create a folder
os.system("rm -rf imgs; rm -Rf video.mp4; mkdir imgs")
imgs = []


# Run a single episode
for num_episode in range(10):
    obs = env.reset()
    print(num_episode)

    done = False
    while not done:
        obs = split_list(obs.reshape((-1,)), 10)
        actions = []
        for ob in obs:
            # action, _states = model.predict(ob, deterministic=True)
            action = ob[7:]
            actions.extend(list(action))
        actions = np.array(actions)
        obs, rewards, done, info = env.step([actions])
        env.render()

        # also save the image to an array
        imgs.append(env.envs[0].renderer.canvas.render())

print("writing images to folder")
for i in range(len(imgs)):
    io.write_png("imgs/{0:05}.png".format(i), imgs[i])

print("To create video, run this command:")
print("ffmpeg -r 60 -i imgs/%05d.png -vcodec libx264 -y -an video.mp4 -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -pix_fmt yuv420p && rm -rf imgs")

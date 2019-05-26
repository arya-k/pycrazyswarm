"""
This script can load an existing model, and create a video
from the frames output by vispy. This can be used for demonstrations.
Any mouse movements / frame movements from the visualizer will be captured
in the video.
"""

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


# Create the environment
env = DummyVecEnv([lambda: DynamicObstEnv()])
model = PPO2.load("Dynamic8")


# Create a folder to store image frames
os.system("rm -rf imgs; rm -Rf video.mp4; mkdir imgs")
counter = 0

# Run some episodes
for num_episode in range(10):  # runs for 10 episodes by default
    obs = env.reset()  # reset the environment; get the first observation
    # sanity check to make sure the script is still running from CLI
    print(num_episode)

    done = False
    while not done:  # only reset when the environment says to
        action, _states = model.predict(
            obs, deterministic=True)  # use model for movements
        obs, reward, done, info = env.step(
            action)  # step model, gather new input
        env.render()  # render the image to the display, to allow for optional mouse input

        # also save the image to an array
        io.write_png("imgs/{0:05}.png".format(counter),
                     env.envs[0].renderer.canvas.render())
        counter += 1

print("Creating video:")
os.system("ffmpeg -r 60 -i imgs/%05d.png -vcodec libx264 -y -an video.mp4 -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -pix_fmt yuv420p && rm -rf imgs")

print("Video file created. See 'video.mp4' in script folder.")

# class RobotPolicy(object):
#     def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, deterministic = False):
#         self.pdtype = make_pdtype(ac_space)
#         with tf.variable_scope("model", reuse=reuse):
#             obs = tf.placeholder(shape=(None, 3), dtype=tf.float32, name="obs")
#             rel_goal = tf.placeholder(shape=(None, 3), dtype=tf.float32, name="rel_goal")
#             velocities = tf.placeholder(shape=(None, 3), dtype=tf.float32, name="velocities")

#             pi_net = self.net(obs, rel_goal, velocities)
#             vf_h2 = self.net(obs, rel_goal, velocities)
#             vf = fc(vf_h2, 'vf', 1)[:,0]

#             self.pd, self.pi = self.pdtype.pdfromlatent(pi_net, init_scale=0.01)

#         if deterministic:
#             a0 = self.pd.mode()
#         else:
#             a0 = self.pd.sample()
#         neglogp0 = self.pd.neglogp(a0)
#         self.initial_state = None

#         self.laser = laser
#         self.rel_goal = rel_goal
#         self.velocities = velocities

#         self.vf = vf

#         def step(ob, *_args, **_kwargs):
#             lb = [o["laser"] for o in ob]
#             rb = [o["rel_goal"] for o in ob]
#             vb = [o["velocities"] for o in ob]

#             #print(rb)

#             a, v, neglogp = sess.run([a0, vf, neglogp0],
#                                      {self.laser: lb, self.rel_goal: rb, self.velocities: vb})
#             return a, v, self.initial_state, neglogp

#         def value(ob, *_args, **_kwargs):
#             lb = [o["laser"] for o in ob]
#             rb = [o["rel_goal"] for o in ob]
#             vb = [o["velocities"] for o in ob]
#             return sess.run(vf, {self.laser: lb, self.rel_goal: rb, self.velocities: vb})

#         self.step = step
#         self.value = value

#     def net(self, laser, rel_goal, velocities):
#         net = tf.layers.conv1d(laser, 32, 5, strides=2, activation=tf.nn.relu)
#         net = tf.layers.conv1d(net, 32, 3, strides=2, activation=tf.nn.relu)
#         net = tf.layers.flatten(net)
#         net = tf.layers.dense(net, 256, activation=tf.nn.relu)


#         net = tf.concat(axis=1, values=[rel_goal, velocities, net])
#         net = tf.layers.dense(net, 256, activation=tf.nn.relu)
#         net = tf.layers.dense(net, 128, activation=tf.nn.relu)
#         net = tf.layers.dense(net, 64, activation=tf.nn.relu)

#         return net

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

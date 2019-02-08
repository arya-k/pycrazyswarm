#!/usr/bin/env python3

"""
This ppo.py file contains the main PPO class, and 
util functions to save / load the class. It also
contains functions to train the PPO class in a multi-
threaded fashion.
"""

import threading
import queue
import os
import cloudpickle
import numpy as np
import tensorflow as tf
from time import time

class Global():

    ''' Globals as a shared object, which can be serialized. '''

    def __init__(self):
        self.UPDATE_EVENT = threading.Event()
        self.ROLLING_EVENT = threading.Event()
        self.GLOBAL_UPDATE_COUNTER = 0
        self.GLOBAL_EP = 0
        self.GLOBAL_RUNNING_R = []
        self.COORD = tf.train.Coordinator()
        self.QUEUE = queue.Queue() # workers put data in this queue
        self.start_t = time()


class PPO():

    ''' Proximal Policy Optimization with Serialization and 
    Multithreading support. Based off of OpenAI Baselines' implementation '''

    def __init__(self, tr_vars):
        ''' tr_vars: the training variables'''
        tf.reset_default_graph()
        self.sess = tf.Session() # create a session
        self.g = Global()
        self.tr_vars = tr_vars
        self.tfs = tf.placeholder( # tensorflow state
            tf.float32,
            [None, tr_vars['ENV'].obs_dim],
            'state')

        # critic
        l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu) # first layer
        self.v = tf.layers.dense(l1, 1) # value
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r') # discounted reward
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage)) # critic loss
        self.ctrain_op = tf.train.AdamOptimizer(
            tr_vars['C_LR']).minimize(self.closs)

        # actor
        pi, pi_params, mu = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params, _ = self._build_anet('oldpi', trainable=False)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0) # sampling action
        self.det_op = tf.squeeze(mu.sample(1), axis=0) # determinstic action
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in 
                                zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, tr_vars['ENV'].act_dim], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(surr, # clipped surrogate objective
            tf.clip_by_value(ratio,
                1. - tr_vars['EPSILON'],
                1. + tr_vars['EPSILON']) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(
            tr_vars['A_LR']).minimize(self.aloss)

        with tf.variable_scope('model'):
              self.params = tf.trainable_variables()

        self.sess.run(tf.global_variables_initializer())

    def update(self):
        ''' Multithreaded training loop. '''
        update_step = self.tr_vars['UPDATE_STEP']
        s_dim = self.tr_vars['ENV'].obs_dim
        a_dim = self.tr_vars['ENV'].act_dim

        while not self.g.COORD.should_stop():
            if self.g.GLOBAL_EP < self.tr_vars['EP_MAX']:
                self.g.UPDATE_EVENT.wait() # wait until get batch of data
                self.sess.run(self.update_oldpi_op) # copy pi to old pi
                data = [self.g.QUEUE.get() for _ in range(self.g.QUEUE.qsize())] # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :s_dim], data[:, s_dim: s_dim + a_dim], data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                # update actor and critic in a update loop
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(update_step)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(update_step)]
                self.g.UPDATE_EVENT.clear() # updating finished
                self.g.GLOBAL_UPDATE_COUNTER = 0 # reset counter
                self.g.ROLLING_EVENT.set() # set roll-out available

    def _build_anet(self, name, trainable=True):
        ''' Build the actor's network. Returns pi and mu. '''
        a_dim = self.tr_vars['ENV'].act_dim # action dimensions
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 200, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, 200, tf.nn.relu, trainable=trainable)
            mu = tf.layers.dense(l2, a_dim, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l2, a_dim, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params, tf.distributions.Normal(loc=mu, scale=0.)

    def sample_action(self, s):
        ''' Get the sampling policy's action, given a state s. '''
        s = s[np.newaxis, :] # reshape the input
        return self.sess.run(self.sample_op, {self.tfs: s})[0]

    def det_action(self, s):
        ''' Get the deterministic policy's action, given a state s. '''
        s = s[np.newaxis, :] # reshape the input
        return self.sess.run(self.det_op, {self.tfs: s})[0]

    def get_v(self, s):
        ''' The critic method's estimate of the reward, given a state s. '''
        if s.ndim < 2: s = s[np.newaxis, :] # reshape input if necessary
        return self.sess.run(self.v, {self.tfs: s})[0, 0] # get critic's output

    def save(self, save_path):
        data = self.tr_vars
        params = self.sess.run(self.params)
        globalvars = None #[self.g.GLOBAL_UPDATE_COUNTER, self.g.GLOBAL_EP, self.g.GLOBAL_RUNNING_R, self.g.]

        if isinstance(save_path, str):
            _, ext = os.path.splitext(save_path)
            if ext == "":
                save_path += ".pkl"
            with open(save_path, "wb") as file_:
                cloudpickle.dump((data, params, globalvars), file_)
        else:
            # Here save_path is a file-like object, not a path
            cloudpickle.dump((data, params, globalvars), save_path)

    @classmethod
    def load(cls, load_path):
        if isinstance(load_path, str):
            if not os.path.exists(load_path):
                if os.path.exists(load_path + ".pkl"):
                    load_path += ".pkl"
                else:
                    raise ValueError("Error: the file {} could not be found".format(load_path))

            with open(load_path, "rb") as file:
                data, params, _ = cloudpickle.load(file)
        else:
            # Here load_path is a file-like object, not a path
            data, params, _ = cloudpickle.load(load_path)

        model = cls(data)

        restores = []
        for param, loaded_p in zip(model.params, params):
            restores.append(param.assign(loaded_p))
        model.sess.run(restores)

        return model


class Worker():

    ''' Multithreaded simulators, which each independently train the
    PPO algorithm, according to it's current policies. '''

    def __init__(self, algo, tr_vars):
        ''' Creates a copy of the env, and shares algo, the PPO object. '''
        self.ppo = algo # global, shared PPO class
        self.env = tr_vars['ENV']() # individual copy of the env.
        self.g = algo.g

        self.ep_len = tr_vars['EP_LEN']
        self.ep_max = tr_vars['EP_MAX']
        self.gamma = tr_vars['GAMMA']
        self.min_batch_size = tr_vars['MIN_BATCH_SIZE']

    def work(self):
        ''' Training loop to optimize PPO while the coordinator states that
        training is not yet complete. Always acts according to the most up to
        date policy.'''
        while not self.g.COORD.should_stop():
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(self.ep_len):
                if not self.g.ROLLING_EVENT.is_set(): # while global PPO is updating
                    self.g.ROLLING_EVENT.wait() # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], [] # clear history buffer, use new policy to collect data
                a = self.ppo.sample_action(s)
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8) # normalize reward, find to be useful
                s = s_
                ep_r += r

                self.g.GLOBAL_UPDATE_COUNTER += 1 # count to minimum batch size, no need to wait other workers
                if t == self.ep_len - 1 or self.g.GLOBAL_UPDATE_COUNTER >= self.min_batch_size:
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = [] # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + self.gamma * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.g.QUEUE.put(np.hstack((bs, ba, br))) # put data in the queue
                    if self.g.GLOBAL_UPDATE_COUNTER >= self.min_batch_size:
                        self.g.ROLLING_EVENT.clear() # stop collecting data
                        self.g.UPDATE_EVENT.set() # globalPPO update

                    if self.g.GLOBAL_EP >= self.ep_max: # stop training
                        self.g.COORD.request_stop()
                        break

            # record reward changes, plot later
            if len(self.g.GLOBAL_RUNNING_R) == 0:
                self.g.GLOBAL_RUNNING_R.append(ep_r)
            else:
                self.g.GLOBAL_RUNNING_R.append(self.g.GLOBAL_RUNNING_R[-1]*0.9+ep_r*0.1)

            print('EP: {} | EP_R: {:.3f} | ETA: {:.0f}m {:.0f}s'.format(
                self.g.GLOBAL_EP, self.g.GLOBAL_RUNNING_R[-1],
                *divmod((time()-self.g.start_t)*(self.ep_max-self.g.GLOBAL_EP-1) /
                    (self.g.GLOBAL_EP + 1), 60))) # VERY ugly print statement ¯\_(ツ)_/¯

            self.g.GLOBAL_EP += 1


def train_multithread(algo, training_vars, num_workers):
    ''' Train algo, a PPO object, using multithreaded workers. '''
    algo.g.UPDATE_EVENT.clear() # not update now
    algo.g.ROLLING_EVENT.set() # start to roll out
    workers = [Worker(algo, training_vars) for _ in range(num_workers)]
    
    threads = []
    for worker in workers: # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start() # training
        threads.append(t)

    # add a PPO updating thread
    threads.append(threading.Thread(target=algo.update,))
    threads[-1].start()
    algo.g.COORD.join(threads)

    return algo.g.GLOBAL_RUNNING_R




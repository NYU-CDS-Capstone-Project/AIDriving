import os
import random

import numpy as np

import gym
from gym.spaces.box import Box

import gym_duckietown
from gym_duckietown.envs import *
from gym_duckietown.wrappers import *
from vec_env.dummy_vec_env import DummyVecEnv
from vec_env.subproc_vec_env import SubprocVecEnv


def make_env(env_id, seed, rank, log_dir, start_container, discrete_wrapper=False):
    def _thunk():
        env = gym.make(env_id)
        if discrete_wrapper:
            env = DiscreteWrapper(env)

        env.seed(seed + rank)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape

        if len(obs_shape) == 3 and obs_shape[2] == 3:
            env = PyTorchObsWrapper(env)

        env = ScaleObservations(env)

        return env

    return _thunk


def make_env_vec(num_processes, env_name, seed, log_dir, start_container, max_steps = 1200, discrete_wrapper=False):
    envs = [make_env(env_name, seed, i, log_dir, start_container, discrete_wrapper)
                for i in range(num_processes)]
    for i in range(num_processes):
        envs[i].max_steps = max_steps

    if num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)
    return envs


class ScaleObservations(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ScaleObservations, self).__init__(env)
        self.obs_lo = self.observation_space.low[0,0,0]
        self.obs_hi = self.observation_space.high[0,0,0]
        obs_shape = self.observation_space.shape
        self.observation_space = Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)

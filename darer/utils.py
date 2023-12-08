import os
import gymnasium as gym
import numpy as np
from stable_baselines3.common.logger import configure
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
import time

import torch
import sys
sys.path.append("../tabular")
sys.path.append("tabular")
from tabular_utils import get_dynamics_and_rewards, solve_unconstrained
from wrappers import FrameStack
# from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gymnasium.wrappers import RecordVideo


def logger_at_folder(log_dir=None, algo_name=None):
    # ensure no _ in algo_name:
    if '_' in algo_name:
        print("WARNING: '_' not allowed in algo_name (used for indexing). Replacing with '-'.")
    algo_name = algo_name.replace('_', '-')
    # Generate a logger object at the specified folder:
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        files = os.listdir(log_dir)
        # Get the number of existing "LogU" directories:
        num = len([int(f.split('_')[1]) for f in files if algo_name in f]) + 1
        tmp_path = f"{log_dir}/{algo_name}_{num}"

        # If the path exists already, increment the number:
        while os.path.exists(tmp_path):
            num += 1
            tmp_path = f"{log_dir}/{algo_name}_{num}"
            # try:
            #     os.makedirs(tmp_path, exist_ok=False)
            # except FileExistsError:
            #     # try again with an incremented number:
            # pass
        logger = configure(tmp_path, ["stdout", "tensorboard"])
    else:
        # print the logs to stdout:
        # , "csv", "tensorboard"])
        logger = configure(format_strings=["stdout"])

    return logger


# class AtariAdapter(gym.Wrapper):
#     """
#     Wrapper for atari-preprocessed environments to make them consistent with unprocessed environments.
#     Observation space has to include a channel dimension.
#     """
#     def __init__(self, env):
#         super().__init__(env)
#         self.env = env
#         obs_space_kwargs = env.observation_space.__dict__
#         obs_space_kwargs['shape'] = (*obs_space_kwargs['_shape'], 1)
#         for key in ['bounded_below', 'bounded_above', '_shape', 'low_repr', 'high_repr', '_np_random']:
#             obs_space_kwargs.pop(key)
#         obs_space_kwargs["low"] = obs_space_kwargs["low"][...,np.newaxis]
#         obs_space_kwargs["high"] = obs_space_kwargs["high"][...,np.newaxis]
#         self.observation_space = gym.spaces.Box(**obs_space_kwargs)
#         self.action_space = env.action_space
#
#     def step(self, action):
#         obs, rew, term, trunk, info = self.env.step(action)
#         obs = obs[..., np.newaxis]
#         return obs, rew, term, trunk, info


# class SaveLastRender(gym.wrapper):
#     def render(mode):
#         if mode=="local":
#             RecordVideo(eval_env, path='video.mp4')


def env_id_to_envs(env_id, render, n_envs, frameskip=1, framestack_k=None, grayscale_obs=False):
    if isinstance(env_id, str):
        # Don't vectorize if there is only one env
        if n_envs==1:
            env = gym.make(env_id, frameskip=1)
            env = AtariPreprocessing(env, screen_size=84, grayscale_obs=grayscale_obs, grayscale_newaxis=True, scale_obs=True, noop_max=30, frame_skip=frameskip)
            if framestack_k:
                env = FrameStack(env, framestack_k)
            # env = AtariAdapter(env)
            # make another instance for evaluation purposes only:
            eval_env = gym.make(env_id, render_mode='human' if render else None, frameskip=1)
            eval_env = AtariPreprocessing(eval_env, screen_size=84, grayscale_obs=grayscale_obs, grayscale_newaxis=True, scale_obs=True, noop_max=30, frame_skip=frameskip)
            if framestack_k:
                eval_env = FrameStack(eval_env, framestack_k)
            # eval_env = AtariAdapter(eval_env)
            # if render:
            #     eval_env = RecordVideo(eval_env, video_folder='videos')
        else:
            env = gym.make_vec(
                env_id, render_mode='human' if render else None, num_envs=n_envs, frameskip=1,
                wrappers=[
                    lambda e: AtariPreprocessing(e, screen_size=84, grayscale_obs=grayscale_obs, grayscale_newaxis=True, scale_obs=True, frame_skip=frameskip, noop_max=30)
                ])

            eval_env = gym.make_vec(
                env_id, render_mode='human' if render else None, num_envs=n_envs, frameskip=1,
                wrappers=[
                    lambda e: AtariPreprocessing(e, screen_size=84, grayscale_obs=grayscale_obs, grayscale_newaxis=True, scale_obs=True, frame_skip=frameskip, noop_max=30)
                ])

    elif isinstance(env_id, gym.Env):
        env = env_id
        # Make a new copy for the eval env:
        import copy
        eval_env = copy.deepcopy(env_id)
    else:
        env = env_id

        # Make a new copy for the eval env:
        import copy
        eval_env = copy.deepcopy(env_id)
        # raise ValueError(
            # "env_id must be a string or gym.Env instance.")

    return env, eval_env

def log_class_vars(self, params):
    logger = self.logger
    for key, value in params.items():
        value = self.__dict__[value]
        # first check if value is a tensor:
        if isinstance(value, torch.Tensor):
            value = value.item()
        logger.record(key, value)

def get_eigvec_values(fa, save_name=None):
    env = fa.env
    nS = env.unwrapped.nS
    nA = fa.nA
    eigvec = np.zeros((nS, nA))
    for i in range(nS):
        eigvec[i, :] = np.mean([logu.forward(i).cpu().detach().numpy() for logu in fa.online_logus.nets],axis=0)

    if save_name is not None:
        np.save(f'{save_name}.npy', eigvec)

    return eigvec

def get_true_eigvec(fa):
    dynamics, rewards = get_dynamics_and_rewards(fa.env.unwrapped)
    # uniform prior:
    n_states, SA = dynamics.shape
    n_actions = int(SA / n_states)
    prior_policy = np.ones((n_states, n_actions)) / n_actions
    solution = solve_unconstrained(
        fa.beta, dynamics, rewards, prior_policy, eig_max_it=1_000_000, tolerance=1e-12)
    l_true, u_true, v_true, optimal_policy, optimal_dynamics, estimated_distribution = solution
    return u_true

def is_tabular(env):
    return isinstance(env.observation_space, gym.spaces.Discrete) and isinstance(env.action_space, gym.spaces.Discrete)
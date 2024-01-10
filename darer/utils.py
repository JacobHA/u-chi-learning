import os
import gymnasium as gym
import numpy as np
from stable_baselines3.common.logger import configure
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
import time

import torch
import sys

import wandb
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
        # another run may be creating a folder:
        time.sleep(0.5)
        num = len([int(f.split('_')[1]) for f in files if algo_name in f]) + 1
        tmp_path = f"{log_dir}/{algo_name}_{num}"

        # If the path exists already, increment the number:
        while os.path.exists(tmp_path):
            # another run may be creating a folder:
            time.sleep(0.5)
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


class PermuteAtariObs(gym.Wrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = self.env
        new_shape = (self.observation_space.shape[-1], *self.observation_space.shape[:-1])
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low.transpose([2,0,1]),
            high=self.observation_space.high.transpose([2,0,1]),
            shape=new_shape,
            dtype=self.observation_space.dtype
        )
        self.action_space = self.action_space

    def step(self, *args, **kwargs):
        res = self.env.step(*args, **kwargs)
        newres = (np.transpose(res[0], [2,1,0]), *res[1:])
        del res
        return newres

    def reset(self, *args, **kwargs):
        res, info = self.env.reset(*args, **kwargs)
        res = np.transpose(res, [2,1,0])
        return res, info


from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
def rllib_env_id_to_envs(env_id, render=False):
    env = gym.make(env_id)
    env = wrap_deepmind(env, framestack=True, noframeskip=False)

    eval_env = gym.make(env_id, render_mode='human' if render else None)
    eval_env = wrap_deepmind(eval_env, framestack=True, noframeskip=False)
    return env, eval_env

def env_id_to_envs(env_id, render, is_atari=False):
    if isinstance(env_id, gym.Env):
        env = env_id
        # Make a new copy for the eval env:
        import copy
        eval_env = copy.deepcopy(env_id)
        return env, eval_env
    if is_atari:
        return atari_env_id_to_envs(env_id, render, n_envs=1, frameskip=4, framestack_k=4)
    else:
        env = gym.make(env_id)
        eval_env = gym.make(env_id, render_mode='human' if render else None)
        return env, eval_env


def atari_env_id_to_envs(env_id, render, n_envs, frameskip=1, framestack_k=None, grayscale_obs=True, permute_dims=False):
    if isinstance(env_id, str):
        # Don't vectorize if there is only one env
        if n_envs==1:
            env = gym.make(env_id, frameskip=frameskip)
            env = AtariPreprocessing(env, terminal_on_life_loss=True, screen_size=84, grayscale_obs=grayscale_obs, grayscale_newaxis=True, scale_obs=False, noop_max=30, frame_skip=1)
            if framestack_k:
                env = FrameStack(env, framestack_k)
            # permute dims for nature CNN in sb3
            if permute_dims:
                env = PermuteAtariObs(env)
            # env = AtariAdapter(env)
            # make another instance for evaluation purposes only:
            eval_env = gym.make(env_id, render_mode='human' if render else None, frameskip=frameskip)
            eval_env = AtariPreprocessing(eval_env, terminal_on_life_loss=True, screen_size=84, grayscale_obs=grayscale_obs, grayscale_newaxis=True, scale_obs=False, noop_max=30, frame_skip=1)
            if framestack_k:
                eval_env = FrameStack(eval_env, framestack_k)
            if permute_dims:
                eval_env = PermuteAtariObs(eval_env)
            # eval_env = AtariAdapter(eval_env)
            # if render:
            #     eval_env = RecordVideo(eval_env, video_folder='videos')
        else:
            env = gym.make_vec(
                env_id, render_mode='human' if render else None, num_envs=n_envs, frameskip=1,
                wrappers=[
                    lambda e: AtariPreprocessing(e, terminal_on_life_loss=True, screen_size=84, grayscale_obs=grayscale_obs, grayscale_newaxis=True, scale_obs=True, frame_skip=frameskip, noop_max=30)
                ])

            eval_env = gym.make_vec(
                env_id, render_mode='human' if render else None, num_envs=n_envs, frameskip=1,
                wrappers=[
                    lambda e: AtariPreprocessing(e, terminal_on_life_loss=True, screen_size=84, grayscale_obs=grayscale_obs, grayscale_newaxis=True, scale_obs=True, frame_skip=frameskip, noop_max=30)
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

def log_class_vars(self, logger, params, use_wandb=False):
    # logger = self.logger
    for key, value in params.items():
        value = self.__dict__[value]
        # first check if value is a tensor:
        if isinstance(value, torch.Tensor):
            value = value.item()
        logger.record(key, value)
        if use_wandb:
            wandb.log({key: value})

def get_eigvec_values(fa, save_name=None):
    env = fa.env
    nS = env.unwrapped.nS
    nA = fa.nA
    eigvec = np.zeros((nS, nA))
    for i in range(nS):

        eigvec[i, :] = np.mean([logu.forward(i).cpu().detach().numpy() for logu in fa.model.nets],axis=0)

    if save_name is not None:
        np.save(f'{save_name}.npy', eigvec)

    # normalize:
    eigvec /= np.linalg.norm(eigvec)
    if save_name is not None:
        np.save(f'{save_name}.npy', eigvec)
    return eigvec

def get_true_eigvec(fa, beta):
    dynamics, rewards = get_dynamics_and_rewards(fa.env.unwrapped)
    # uniform prior:
    n_states, SA = dynamics.shape
    n_actions = int(SA / n_states)
    prior_policy = np.ones((n_states, n_actions)) / n_actions
    solution = solve_unconstrained(
        beta, dynamics, rewards, prior_policy, eig_max_it=1_000_000, tolerance=1e-12)
    l_true, u_true, v_true, optimal_policy, optimal_dynamics, estimated_distribution = solution
    
    # normalize:
    u_true /= np.linalg.norm(u_true)
    return u_true

def is_tabular(env):
    return isinstance(env.observation_space, gym.spaces.Discrete) and isinstance(env.action_space, gym.spaces.Discrete)
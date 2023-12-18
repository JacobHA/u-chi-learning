import os
import gymnasium as gym
import numpy as np
from stable_baselines3.common.logger import configure
import time
import wandb

import torch
import sys

import wandb
sys.path.append("tabular")
from tabular_utils import get_dynamics_and_rewards, solve_unconstrained


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

from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
def rllib_env_id_to_envs(env_id, render=False):
    env = gym.make(env_id)
    env = wrap_deepmind(env, framestack=True, noframeskip=False)

    eval_env = gym.make(env_id, render_mode='human' if render else None)
    eval_env = wrap_deepmind(eval_env, framestack=True, noframeskip=False)
    return env, eval_env


def env_id_to_envs(env_id, render):
    if isinstance(env_id, str):
        env = gym.make(env_id)
        # make another instance for evaluation purposes only:
        eval_env = gym.make(env_id, render_mode='human' if render else None)
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

def log_class_vars(self, params, use_wandb=False):
    logger = self.logger
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
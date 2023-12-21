import json
import sys

from stable_baselines3 import DQN
sys.path.append("darer")
from utils import env_id_to_envs, rllib_env_id_to_envs
import numpy as np
import gymnasium as gym
env_id = 'PongNoFrameskip-v4'
env, eval_env = rllib_env_id_to_envs(env_id, render=False)

from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecMonitor

with open("experiments/baselines/dqn-nature.json", "r") as f:
    hparams = json.load(f)

total_steps = 10_000_000
# rename to match stable baselines3:
hparams['gamma'] = hparams.pop('discount_factor')
hparams['exploration_fraction'] = hparams.pop('exploration_final_eps_frame') / total_steps
n_stack = hparams.pop("action_history_len")
# remove unused hparams
hparams.pop('gradient_momentum')
hparams.pop('squared_gradient_momentum')
hparams.pop('min_squared_gradient')
hparams.pop('action_repeat')
# dummy vectorize:
# env = DummyVecEnv([lambda: env])
# env = VecFrameStack(env, n_stack=n_stack)
# env = VecMonitor(env)
model = DQN('CnnPolicy', env, verbose=4, device='cuda',
            policy_kwargs={'normalize_images': True}, **hparams)

model.learn(total_timesteps=total_steps, log_interval=3)
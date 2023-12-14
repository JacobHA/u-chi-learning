from stable_baselines3 import DQN
from utils import env_id_to_envs, rllib_env_id_to_envs
import numpy as np
import gymnasium as gym
env_id = 'PongNoFrameskip-v4'
env, eval_env = rllib_env_id_to_envs(env_id,
                    render=False,
)

from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecMonitor

# env = AtariAdapter(env)
from hparams import pong_logu
hparams = pong_logu
hparams.pop('beta')
hparams.pop('theta_update_interval')
hparams.pop('tau_theta')
hparams.pop('hidden_dim')
hparams['buffer_size'] = 50000
# dummy vectorize:
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)
env = VecMonitor(env)
model = DQN('CnnPolicy', env, verbose=4, device='cuda',
            policy_kwargs={'normalize_images': False}, **hparams)

model.learn(total_timesteps=1_000_000, log_interval=3)
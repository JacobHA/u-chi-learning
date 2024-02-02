import sys

import torch.nn.functional
import yaml

sys.path.append('darer')
import argparse
from CustomDQN import CustomDQN
from UAgent import UAgent
from SoftQAgent import SoftQAgent
from hparams import *
import time


str_to_algo = {
    'u': UAgent,
    'dqn': CustomDQN,
    'sql': SoftQAgent
}


env_id_to_timesteps = {
    'CartPole-v1': 50_000,
    'Acrobot-v1': 50_000,
    'LunarLander-v2': 500_000,
    'PongNoFrameskip-v4': 10_000_000,
    'MountainCar-v0': 120_000,
}

loss_str_to_fn = {
    "smooth_l1": torch.nn.functional.smooth_l1_loss,
}

def runner(env_id, algo_str, device, tensorboard_log, config=None, total_timesteps=None):
    # for local run debugging
    if not config:
        try:
            config = id_to_hparam_dicts[env_id][algo_str]
        except KeyError:
            raise ValueError(f"env {env_id} not recognized.")
        # hardcode some of the parameters of interest
        if algo_str == 'u' or algo_str == 'logu':
            rawlik_hparams = {
                'use_rawlik': False,
                'prior_update_interval': 1_000,
                'prior_tau': 0.90,
            }
            config.update(rawlik_hparams)

    algo = str_to_algo[algo_str]
    model = algo(env_id, **config, tensorboard_log=tensorboard_log,
                 device=device, log_interval=10000)#, aggregator='max')
    model.learn(total_timesteps=total_timesteps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--count', type=int, default=10)
    parser.add_argument('-a', '--algo', type=str, default='dqn')
    parser.add_argument('-e', '--env', type=str, default="MountainCar-v0")  # 'LunarLander-v2'
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('--config', type=str, default=None)  # sweep_params/u-lunar-unstable-fine.yml

    args = parser.parse_args()
    env = args.env

    start = time.time()
    tensorboard_log = f'experiments/ft/{args.env}'
    total_timesteps = env_id_to_timesteps[args.env]
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.load(f, yaml.SafeLoader)
        env = config.pop('env_id')
        config['loss_fn'] = loss_str_to_fn[config['loss_fn']]
        config['learning_starts'] = total_timesteps * config.pop('learning_starts_ratio')
    for i in range(args.count):
        if args.algo == '*':
            for algo in str_to_algo.keys():
                runner(args.env, algo, args.device, tensorboard_log, total_timesteps=total_timesteps, config=config)
        else:
            runner(args.env, args.algo, args.device, tensorboard_log, total_timesteps=total_timesteps, config=config)
        print(f"Finished run {i+1}/{args.count}")
    print(f"trained in {time.time() - start}")

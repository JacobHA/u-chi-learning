import sys

import torch.nn.functional

sys.path.append('darer')
import argparse
from CustomDQN import CustomDQN
from CustomPPO import CustomPPO
# from LogU import LogULearner
from UAgent import UAgent
from SoftQAgent import SoftQAgent
from hparams import *
import time

# env = 'CartPole-v1'
# env = 'LunarLander-v2'
# env = 'Acrobot-v1'
# env = 'MountainCar-v0'

str_to_algo = {
    'u': UAgent,
    'u-norwl': UAgent,
    # 'logu': LogUAgent,
    'ppo': CustomPPO,
    'dqn': CustomDQN,
    'sql': SoftQAgent
}


env_id_to_timesteps = {
    'CartPole-v1': 50_000,
    'Acrobot-v1': 50_000,
    'LunarLander-v2': 500_000,
    'PongNoFrameskip-v4': 1_000_000,
    'MountainCar-v0': 100_000,
}

def runner(env_id, algo_str, device, tensorboard_log, config=None, total_timesteps=None):
    # for local run debugging
    if not config:
        if env_id == 'MountainCar-v0':
            configs = mcars
        elif env_id == 'CartPole-v1':
            configs = cartpoles
        elif env_id == 'LunarLander-v2':
            configs = lunars
        elif env_id == 'Acrobot-v1':
            configs = acrobots
        else:
            raise ValueError(f"env {env_id} not recognized.")
        # Now access the config for this algo:
        config = configs[algo_str]
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
    parser.add_argument('-c', '--count', type=int, default=15)
    parser.add_argument('-a', '--algo', type=str, default='ppo')
    # 'CartPole-v1'
    # 'LunarLander-v2'
    # 'Acrobot-v1'
    # 'MountainCar-v0'
    parser.add_argument('-e', '--env', type=str, default='LunarLander-v2')
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-p', '--ppi', type=bool, default=False)

    args = parser.parse_args()
    env = args.env


    start = time.time()
    tensorboard_log = f'experiments/ft/{args.env}'
    total_timesteps = env_id_to_timesteps[args.env]
    try:
        config = id_to_hparam_dicts[env][algo]
    except KeyError:
        raise ValueError(f"env {env} not recognized.")
    for i in range(args.count):
        if args.algo == '*':
            for algo in str_to_algo.keys():
                runner(args.env, algo, args.device, tensorboard_log, total_timesteps=total_timesteps)
        else:
            runner(args.env, args.algo, args.device, tensorboard_log, total_timesteps=total_timesteps)
        print(f"Finished run {i+1}/{args.count}")
    print(f"trained in {time.time() - start}")

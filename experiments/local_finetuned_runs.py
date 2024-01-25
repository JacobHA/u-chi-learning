import sys
sys.path.append('darer')
import argparse
from CustomDQN import CustomDQN
from CustomPPO import CustomPPO
# from LogU import LogULearner
from UAgent import UAgent
from LogUAgent import LogUAgent
from hparams import *
import time


str_to_algo = {
    'u': UAgent,
    'u-norwl': UAgent,
    # 'logu': LogUAgent,
    'ppo': CustomPPO,
    'dqn': CustomDQN
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
                 device=device, log_interval=1000)#, aggregator='max')
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

    args = parser.parse_args()

    start = time.time()
    tensorboard_log = f'experiments/ft/{args.env}'
    total_timesteps = 250_000
    for i in range(args.count):
        if args.algo == '*':
            for algo in str_to_algo.keys():
                runner(args.env, algo, args.device, tensorboard_log, total_timesteps=total_timesteps)
        else:
            runner(args.env, args.algo, args.device, tensorboard_log, total_timesteps=total_timesteps)
        print(f"Finished run {i+1}/{args.count}")
    print(f"trained in {time.time() - start}")

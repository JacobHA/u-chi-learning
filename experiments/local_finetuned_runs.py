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
    # 'logu': LogUAgent,
    'ppo': CustomPPO,
    'dqn': CustomDQN
}

def runner(env, algo_str, device):
    if env == 'MountainCar-v0':
        configs = mcars
    elif env == 'CartPole-v1':
        configs = cartpoles
    elif env == 'LunarLander-v2':
        configs = lunars
    elif env == 'Acrobot-v1':
        configs = acrobots
    else:
        raise ValueError(f"env {env} not recognized.")

    # Now access the config for this algo:
    config = configs[algo_str]
    algo = str_to_algo[algo_str]

    rawlik_hparams = {
        'use_rawlik': True,
        'prior_update_interval': 1_000,
        'prior_tau': 0.90,
    }
    if algo_str == 'u' or algo_str == 'logu':
        config.update(rawlik_hparams)

    model = algo(env, **config, tensorboard_log=f'experiments/ft/{env}',
                 device=device, log_interval=1000)#, aggregator='max')
    model.learn(total_timesteps=250_000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--count', type=int, default=15)
    parser.add_argument('-a', '--algo', type=str, default='*')
    # 'CartPole-v1'
    # 'LunarLander-v2'
    # 'Acrobot-v1'
    # 'MountainCar-v0'
    parser.add_argument('-e', '--env', type=str, default='LunarLander-v2')
    parser.add_argument('-d', '--device', type=str, default='cuda')

    args = parser.parse_args()

    start = time.time()
    for i in range(args.count):
        if args.algo == '*':
            for algo in str_to_algo.keys():
                runner(args.env, algo, args.device)
        else:
            runner(args.algo, args.device)
        print(f"Finished run {i+1}/{args.count}")
    print(f"trained in {time.time() - start}")

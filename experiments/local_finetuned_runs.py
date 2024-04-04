import sys
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
    'ppo': CustomPPO,
    'dqn': CustomDQN,
    'sql': SoftQAgent
}


env_id_to_timesteps = {
    'CartPole-v1': 50_000,
    'Acrobot-v1': 50_000,
    'LunarLander-v2': 100_000,
    'PongNoFrameskip-v4': 10_000_000,
    'MountainCar-v0': 500_000,
}

env_to_log_interval = {
    'CartPole-v1': 1000,
    'Acrobot-v1': 1000,
    'LunarLander-v2': 1000,
    'MountainCar-v0': 1000,
}


def runner(algo, device):
    try:
        config = id_to_hparam_dicts[env][algo]
    except KeyError:
        raise ValueError(f"env {env} not recognized.")

    # Now access the config for this algo:
    algo = str_to_algo[algo]

    rawlik_hparams = {'use_rawlik': ppi}

    if algo == UAgent:
        model = algo(env, **config, tensorboard_log=f'experiments/ft/{env}',
                 device=device, log_interval=env_to_log_interval[env],
                  **rawlik_hparams)#, name='U-rawlik2')#,
    else:
        model = algo(env, **config, tensorboard_log=f'experiments/ft/{env}',
                 device=device, log_interval=env_to_log_interval[env])
        
    total_timesteps = env_id_to_timesteps[env]
    model.learn(total_timesteps=total_timesteps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--count', type=int, default=5)
    parser.add_argument('-a', '--algo', type=str, default='u')
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-e', '--env', type=str, default='MountainCar-v0')
    parser.add_argument('-p', '--ppi', type=bool, default=False)


    args = parser.parse_args()
    env = args.env
    ppi = args.ppi


    start = time.time()
    for i in range(args.count):
        runner(args.algo, args.device)
        print(f"Finished run {i+1}/{args.count}")
    print(f"trained in {time.time() - start}")

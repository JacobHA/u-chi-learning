import sys
sys.path.append('darer')
import argparse
from CustomDQN import CustomDQN
from CustomPPO import CustomPPO
# from LogU import LogULearner
from UAgent import UAgent
from hparams import *
import time

# env = 'CartPole-v1'
# env = 'LunarLander-v2'
# env = 'Acrobot-v1'
# env = 'MountainCar-v0'

str_to_algo = {
    'u': UAgent,
    'ppo': CustomPPO,
    'dqn': CustomDQN
}


env_id_to_timesteps = {
    'CartPole-v1': 50_000,
    'Acrobot-v1': 50_000,
    'LunarLander': 250_000,
    'PongNoFrameskip-v4': 1_000_000,
    'MountainCar-v0': 100_000,
}


def runner(algo, device):
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
    config = configs[algo]
    algo = str_to_algo[algo]

    rawlik_hparams = {'use_rawlik': True,
                    'prior_update_interval': 5000,
                    'prior_tau': 0.995,
                        }

    model = algo(env, **config, tensorboard_log=f'experiments/ft/{env}',
                 device=device, log_interval=1000, **rawlik_hparams)#, name='irred')#, 
    total_timesteps = env_id_to_timesteps[env]
    model.learn(total_timesteps=total_timesteps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--count', type=int, default=5)
    parser.add_argument('-a', '--algo', type=str, default='u')
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-e', '--env', type=str, default='MountainCar-v0')

    args = parser.parse_args()
    env = args.env


    start = time.time()
    for i in range(args.count):
        runner(args.algo, args.device)
        print(f"Finished run {i+1}/{args.count}")
    print(f"trained in {time.time() - start}")

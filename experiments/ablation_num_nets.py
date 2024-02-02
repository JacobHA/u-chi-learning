import sys
sys.path.append('darer')
import argparse
from CustomDQN import CustomDQN
from CustomPPO import CustomPPO
# from LogU import LogULearner
from UAgent import UAgent
from hparams import id_to_hparam_dicts, configs
import time

# env = 'CartPole-v1'
env = 'Acrobot-v1'
# env = 'MountainCar-v0'


def runner(device):
    try:
        config = id_to_hparam_dicts[env]['u']
    except KeyError:
        print(f"env {env} not recognized.")

    # Now access the config for this algo:
    config = configs['u']

    rawlik_hparams = {'use_rawlik': False}

    model = UAgent(env, **config, tensorboard_log=f'experiments/ablations/{env}',
                 device=device, log_interval=1000, **rawlik_hparams,
                 name=f'{NUM_NETS}nets',
                 num_nets=NUM_NETS
                 )#, aggregator='max')
    model.learn(total_timesteps=50_000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--count', type=int, default=5)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-n', '--num_nets', type=int, default=2)
    args = parser.parse_args()

    NUM_NETS = args.num_nets

    start = time.time()
    for i in range(args.count):
        runner(args.device)
        print(f"Finished run {i+1}/{args.count}")
    print(f"trained in {time.time() - start}")

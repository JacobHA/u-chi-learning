import argparse
import yaml
import sys

sys.path.append('darer')
from SoftQAgent import SoftQAgent
from CustomDQN import CustomDQN
from UAgent import UAgent
from arSAC import arSAC
from LogUAgent import LogUAgent
from utils import safe_open


env_to_steps = {
    'CartPole-v1': 10_000,
    'Acrobot-v1': 100_000,
    'LunarLander-v2': 200_000,
    'MountainCar-v0': 500_000,
    'HalfCheetah-v4': 1_000_000,
    'Ant-v4': 1_000_000,
}

env_to_logfreq = {
    'CartPole-v1': 200,
    'Acrobot-v1': 200,
    'LunarLander-v2': 1000,
    'MountainCar-v0': 100,
    'HalfCheetah-v4': 2500,
}

args = argparse.ArgumentParser()
args.add_argument('--count', type=int, default=10)
args.add_argument('--env_id', type=str, default='HalfCheetah-v4')
args.add_argument('--algo', type=str, default='arSAC')
args.add_argument('--device', type=str, default='auto')
args.add_argument('--exp-name', type=str, default='EVAL')

args = args.parse_args()
env_id = args.env_id
experiment_name = args.exp_name
device = args.device

print("Running finetuned hyperparameters...")
algo = args.algo
print(algo)

hparams = safe_open(f'hparams/{env_id}/{algo}.yaml')
# Drop the gamma hparam:
if algo == 'u': 
    try:
        hparams.pop('gamma')
    except:
        pass
    AgentClass = UAgent
elif algo == 'dqn':
    AgentClass = CustomDQN
elif algo == 'sql':
    AgentClass = SoftQAgent
elif algo == 'arSAC':
    AgentClass = arSAC
elif algo == 'logu':
    AgentClass = LogUAgent

for i in range(args.count):
    full_config = {}
    # with open(f'hparams/{env_id}/{algo}.yaml') as f:
    #     default_params = yaml.load(f, yaml.FullLoader)
    # #  = yaml.load(open(f'hparams/{env_id}/{algo}.yaml'), yaml.FullLoader)
    # full_config.update(hparams)
    # full_config.update(default_params)

    agent = AgentClass(env_id, **hparams,
                        device=device, log_interval=env_to_logfreq.get(env_id, 1000),
                        tensorboard_log=f'ft_logs/{experiment_name}/{env_id}',
                        render=False,
                        )

    # Measure the time it takes to learn: 
    agent.learn(total_timesteps=env_to_steps.get(env_id, 100_000))
    del agent
import argparse
import yaml
import sys

sys.path.append('darer')
from SoftQAgent import SoftQAgent
from CustomDQN import CustomDQN
from CustomSAC import CustomSAC
from UAgent import UAgent
from ASAC import ASAC
from ASQL import ASQL
from arDDPG import arDDPG
from utils import safe_open

from stable_baselines3.common.callbacks import CheckpointCallback


env_to_steps = {
    'CartPole-v1': 50_000,
    'Acrobot-v1': 50_000,
    'LunarLander-v2': 200_000,
    'MountainCar-v0': 240_000,
    'HalfCheetah-v4': 1_000_000,
    'Ant-v4': 1_000_000,
    'Swimmer-v4': 1_000_000,
    'Humanoid-v4': 5_000_000,
    'Pusher-v4': 1_000_000,
    'Pendulum-v1': 10_000,
    'PongNoFrameskip-v4': 2_000_000,
    'AsterixNoFrameskip-v4': 2_000_000,
    'AlienNoFrameskip-v4': 2_000_000,
    'BreakoutNoFrameskip-v4': 2_000_000,
}

env_to_logfreq = {
    'CartPole-v1': 200,
    'Acrobot-v1': 200,
    'LunarLander-v2': 1000,
    'MountainCar-v0': 1000,
    'HalfCheetah-v4': 2500,
    'Swimmer-v4': 5000,
    'Ant-v4': 5000,
    'Humanoid-v4': 10000,
    'Pusher-v4': 5000,
    'Pendulum-v1': 200,
    'PongNoFrameskip-v4': 10_000,
    'AsterixNoFrameskip-v4': 10_000,
    'AlienNoFrameskip-v4': 10_000,
    'BreakoutNoFrameskip-v4': 10_000,
}

cnnpolicy_envs = { 'PongNoFrameskip-v4', 'AsterixNoFrameskip-v4', 'AlienNoFrameskip-v4', 'BreakoutNoFrameskip-v4'}

args = argparse.ArgumentParser()
args.add_argument('--count', type=int, default=30)
args.add_argument('--env_id', type=str, default='AsterixNoFrameskip-v4')
args.add_argument('--algo', type=str, default='asql')
args.add_argument('--device', type=str, default='auto')
args.add_argument('--exp-name', type=str, default='EVAL')
args.add_argument('--name', type=str, default='')
args.add_argument('--eval_steps', type=int, default=None)

args = args.parse_args()
env_id = args.env_id
experiment_name = args.exp_name
device = args.device
name_suffix = args.name

print("Running finetuned hyperparameters...")
algo = args.algo
algo = algo.lower()
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
    hparams.pop('total_timesteps')
    hparams.pop('log_freq')
    if env_id in cnnpolicy_envs:
        hparams['policy'] = "CnnPolicy"
elif algo == 'sql':
    AgentClass = SoftQAgent
elif algo == 'asac':
    AgentClass = ASAC
elif algo == 'asql':
    AgentClass = ASQL
elif algo == 'sac':
    AgentClass = CustomSAC
elif algo == 'arddpg':
    AgentClass = arDDPG

for i in range(args.count):
    full_config = {}
    logdir = f'ft_logs/{experiment_name}/{env_id}/'
    from stable_baselines3.sac import SAC
    # agent = SAC('MlpPolicy', env_id, **hparams, device=device)
    agent = AgentClass(env_id, **hparams,
                        device=device, log_interval=env_to_logfreq.get(env_id, 1000),
                        tensorboard_log=logdir,
                        max_eval_steps=args.eval_steps,
                        name_suffix=f'{name_suffix}',
                        )
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=agent.logger.get_dir(),
        name_prefix="ft",
    )
    # Measure the time it takes to learn: 
    agent.learn(total_timesteps=env_to_steps.get(env_id, 100_000), callback=checkpoint_callback)
    del agent
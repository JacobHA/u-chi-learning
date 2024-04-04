import argparse
import yaml
import sys

sys.path.append('darer')
from SoftQAgent import SoftQAgent
from CustomDQN import CustomDQN
from UAgent import UAgent
from dm_control import suite


env_to_steps = {
    'CartPole-v1': 10_000,
    'Acrobot-v1': 5_000,
    'LunarLander-v2': 200_000,
    'MountainCar-v0': 500_000,
}

env_to_logfreq = {
    'CartPole-v1': 200,
    'Acrobot-v1': 200,
    'LunarLander-v2': 1000,
    'MountainCar-v0': 100,
}

args = argparse.ArgumentParser()
args.add_argument('--count', type=int, default=10)
args.add_argument('--env_id', type=str, default='Acrobot-v1')
args.add_argument('--algo', type=str, default='u')
args.add_argument('--device', type=str, default='cpu')
args.add_argument('--exp-name', type=str, default='EVAL')

args = args.parse_args()
env_id = args.env_id
if 'dmc:' in env_id:
    env_id = env_id.split(':')[-1]
    domain_name, task_name, _ = env_id.split('-')
    # e.g. dmc:cartpole-swingup-v1
    env_id = suite.load(domain_name=domain_name, task_name=task_name)

experiment_name = args.exp_name
device = args.device

print("Running finetuned hyperparameters...")
algo = args.algo
print(algo)

hparams = yaml.safe_load(open(f'hparams/{env_id}/{algo}.yaml'))
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

for i in range(args.count):
    full_config = {}
    default_params = yaml.safe_load(open(f'hparams/{env_id}/{algo}.yaml'))
    full_config.update(hparams)
    full_config.update(default_params)

    agent = AgentClass(env_id, **full_config,
                        device='auto', log_interval=env_to_logfreq[env_id],
                        tensorboard_log=f'ft_logs/{experiment_name}/{env_id}', num_nets=2,
                        render=False,
                        )

    # Measure the time it takes to learn: 
    agent.learn(total_timesteps=env_to_steps[env_id])
    del agent
import argparse
import wandb
import sys
import torch
import os
import yaml

sys.path.append('darer')

from CustomDQN import CustomDQN
from SoftQAgent import SoftQAgent
from UAgent import UAgent


env_to_total_timesteps = {
    "CartPole-v1": 1_000_000,
    "MountainCar-v0": 1_000_000,
    "LunarLander-v2": 250_000,
    "Acrobot-v1": 1_000_000,
}
loss_str_to_fn = {
    "smooth_l1": torch.nn.functional.smooth_l1_loss,
}

int_hparams = {'batch_size', 'buffer_size', 'gradient_steps',
               'target_update_interval', 'theta_update_interval',
               'train_freq', 'gradient_steps'}
device = None
experiment_name = None
env_id = None


def wandb_train():
    """:param local_cfg: pass config sweep if running locally to sample without wandb"""
    wandb_kwargs = {"group": experiment_name}
    for i in range(n_hparam_runs):
        with wandb.init(**wandb_kwargs, sync_tensorboard=True) as run:
            config = wandb.config.as_dict()
            global env_id
            try:
                env_id = config.pop('env_id')
                config.pop('algo_name')
            except:
                pass
            total_timesteps = env_to_total_timesteps[env_id]
            config['loss_fn'] = torch.nn.functional.mse_loss
            algo_str = config.pop('algo_name')
            if algo_str == 'u':
                config.pop('gamma')
                AgentClass = UAgent
            elif algo_str == 'dqn':
                AgentClass = CustomDQN
            elif algo_str == 'sql':
                AgentClass = SoftQAgent
            agent = AgentClass(env_id, **config,
                               device='cpu', log_interval=200,
                               tensorboard_log='ft_logs', num_nets=2,
                               render=False)

            # Measure the time it takes to learn:
            agent.learn(total_timesteps=total_timesteps)
            del agent


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--sweep", type=str, default=None)
    args.add_argument("--n_runs", type=int, default=100)
    args.add_argument("--proj", type=str, default="u-chi-learning")
    args.add_argument("--local-wandb", type=bool, default=True)
    args.add_argument("--device", type=str, default='cuda')
    args.add_argument("--n-hparam-runs", type=int, default=3, help="number of times to re-train with a single set of hyperparameters")
    args.add_argument("--entity", type=str, default=None)
    args = args.parse_args()
    project = args.proj
    envs = os.listdir('hparams')
    for env in envs:
        if env in args.sweep_file:
            env_id = env
            break
    experiment_name = args.sweep_file.split('/')[1]
    experiment_name = experiment_name[:experiment_name.index(env_id)-1]
    n_hparam_runs = args.n_hparam_runs
    device = args.device
    print(f"continuing sweep {args.sweep}")
    wandb.agent(args.sweep, project=args.proj, count=args.n_runs, function=wandb_train)

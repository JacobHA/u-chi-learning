import sys
sys.path.append('darer')

import argparse
import wandb
import yaml
import numpy as np
import random
import copy
from UAgent import main


from utils import sample_wandb_hyperparams


exp_to_config = {
    # all of the 106 atari environments + hyperparameters. This will take a long time to train.
    "atari-v0": "logu-atari-full-sweep.yml",
    # three of the atari environments
    "atari-mini": "logu-atari-mini-sweep.yml",
    # pong only:
    "atari-pong": "logu-atari-pong-sweep.yml"
}
int_hparams = {'batch_size', 'buffer_size', 'gradient_steps',
               'target_update_interval', 'theta_update_interval'}
device = None


def get_sweep_config(sweepcfg, default_config, project_name):
    cfg = default_config
    params = cfg['parameters']
    params.update(sweepcfg['parameters'])
    cfg.update(sweepcfg)
    cfg['parameters'] = params
    cfg['name'] = project_name
    return cfg


def wandb_train(local_cfg=None):
    """:param local_cfg: pass config sweep if running locally to sample without wandb"""
    wandb_kwargs = {"project": project, "group": experiment_name}
    if local_cfg:
        local_cfg["controller"] = {'type': 'local'}
        sampled_params = sample_wandb_hyperparams(local_cfg["parameters"], int_hparams=int_hparams)
        local_cfg["parameters"] = sampled_params
        print(f"locally sampled params: {sampled_params}")
        wandb_kwargs['config'] = local_cfg
    with wandb.init(**wandb_kwargs, sync_tensorboard=True) as run:
        config = wandb.config.as_dict()
        main(total_timesteps=1_200_000, n_envs=1, log_dir='local-pong',
             device=device, render=False, **config['parameters'])


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--sweep", type=str, default=None)
    args.add_argument("--n_runs", type=int, default=100)
    args.add_argument("--proj", type=str, default="u-chi-learning-test")
    args.add_argument("--local-wandb", type=bool, default=True)
    args.add_argument("--exp-name", type=str, default="atari-pong")
    args.add_argument("--device", type=str, default='cuda')
    args = args.parse_args()
    project = args.proj
    experiment_name = args.exp_name
    device = args.device
    # load the default config
    with open("sweep_params/logu-atari-default.yml", "r") as f:
        default_config = yaml.load(f, yaml.SafeLoader)
    # load the experiment config
    with open(f"sweep_params/{exp_to_config[experiment_name]}", "r") as f:
        expsweepcfg = yaml.load(f, yaml.SafeLoader)
    # combine the two
    sweepcfg = get_sweep_config(expsweepcfg, default_config, project)
    # generate a new sweep if one was not passed as an argument
    if args.sweep is None and not args.local_wandb:
        sweep_id = wandb.sweep(sweepcfg, project=project)
        print(f"created new sweep {sweep_id}")
        wandb.agent(sweep_id, project=args.proj,
                    count=args.n_runs, function=wandb_train)
    elif args.local_wandb:
        for i in range(args.n_runs):
            try:
                print(f"running local sweep {i}")
                wandb_train(local_cfg=copy.deepcopy(sweepcfg))
            except Exception as e:
                print(f"failed to run local sweep {i}")
                print(e)
    else:
        print(f"continuing sweep {args.sweep}")
        wandb.agent(args.sweep, project=args.proj,
                    count=args.n_runs, function=wandb_train)

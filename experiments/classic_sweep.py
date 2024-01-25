import argparse
import wandb
import yaml
import copy
import sys
import traceback as tb
sys.path.append('darer')

from utils import sample_wandb_hyperparams
from local_finetuned_runs import runner


algo_to_config = {
    "ppo": "ppo-classic.yml",
    "u": "u-classic.yml",
    "u-norwl": "u-classic-norwl.yml",  # rawlik disabled
}
env_to_total_timesteps = {
    "CartPole-v1": 1_000_000,
    "MountainCar-v0": 1_000_000,
    "LunarLander-v2": 250_000,
    "Acrobot-v1": 1_000_000,
}

int_hparams = {'batch_size', 'buffer_size', 'gradient_steps',
               'target_update_interval', 'theta_update_interval'}
device = None
experiment_name = None
algo_str = None


def get_sweep_config(sweepcfg, default_config, project_name):
    cfg = default_config
    params = cfg['parameters']
    params.update(sweepcfg['parameters'])
    cfg.update(sweepcfg)
    cfg['parameters'] = params
    cfg['name'] = project_name
    return cfg


def wandb_train(local_cfg=None, env_id=None, n_hparam_runs=None):
    """:param local_cfg: pass config sweep if running locally to sample without wandb"""
    wandb_kwargs = {"project": project, "group": experiment_name}
    if local_cfg:
        local_cfg["controller"] = {'type': 'local'}
        sampled_params = sample_wandb_hyperparams(local_cfg["parameters"], int_hparams=int_hparams)
        local_cfg["parameters"] = sampled_params
        print(f"locally sampled params: {sampled_params}")
        wandb_kwargs['config'] = local_cfg
    for i in range(n_hparam_runs):
        with wandb.init(**wandb_kwargs, sync_tensorboard=True) as run:
            config = wandb.config.as_dict()
            if not env_id:
                env_id = config['parameters'].pop('env_id')
            learn_ratio = config['parameters'].pop('learning_starts_ratio')
            total_timesteps = env_to_total_timesteps[env_id]
            config['parameters']['learning_starts'] = total_timesteps * learn_ratio
            runner(env_id, algo_str, device,
                   tensorboard_log=f'local-{algo_str}-{env_id}',
                   config=config['parameters'],
                   total_timesteps=total_timesteps)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--sweep", type=str, default=None)
    args.add_argument("--n_runs", type=int, default=100)
    args.add_argument("--proj", type=str, default="u-chi-learning-test")
    args.add_argument("--local-wandb", type=bool, default=True)
    args.add_argument("--exp-name", type=str, default="classic-bench")
    args.add_argument("--algo", type=str, default="u-norwl")
    args.add_argument("--device", type=str, default='cuda')
    args.add_argument("--n-hparam-runs", type=int, default=5, help="number of times to re-train with a single set of hyperparameters")
    args.add_argument("--env_id", type=str, default=None, help="env id to run. If none, will sweep over envs in the experiment config")
    args = args.parse_args()
    project = args.proj
    algo_str = args.algo
    experiment_name = args.exp_name
    n_hparam_runs = args.n_hparam_runs
    env_id = args.env_id
    device = args.device
    with open(f"sweep_params/{algo_to_config[algo_str]}", "r") as f:
        expsweepcfg = yaml.load(f, yaml.SafeLoader)
    # combine the two
    sweepcfg = get_sweep_config(expsweepcfg, expsweepcfg, project)
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
                wandb_train(local_cfg=copy.deepcopy(sweepcfg), env_id=env_id, n_hparam_runs=n_hparam_runs)
            except Exception as e:
                print(f"failed to run local sweep {i}")
                tb.print_tb(e.__traceback__)
    else:
        print(f"continuing sweep {args.sweep}")
        wandb.agent(args.sweep, project=args.proj,
                    count=args.n_runs, function=wandb_train)

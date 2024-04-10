import hashlib
import json

import gymnasium
import wandb
import argparse
import yaml
import sys
import os

sys.path.append('darer')
from UAgent import UAgent
from LogUAgent import LogUAgent
from arSAC import arSAC
from utils import safe_open, sample_wandb_hyperparams


env_to_steps = {
    'CartPole-v1': 10_000,
    'Acrobot-v1': 5_000,
    'LunarLander-v2': 200_000,
    'MountainCar-v0': 500_000,
    'HalfCheetah-v4': 1_000_000,
}

env_to_logfreq = {
    'CartPole-v1': 200,
    'Acrobot-v1': 200,
    'LunarLander-v2': 1000,
    'MountainCar-v0': 100,
    'HalfCheetah-v4': 2500,
}

algo_to_agent = {
    'u': UAgent,
    'arSAC': arSAC,
    'logu': LogUAgent
}

int_hparams = {'train_freq', 'gradient_steps'}

SEMAPHORE_FN = "creating_sweep"
SWEEP_ID_FN = "sweep_id"
env_id = 'Acrobot-v1'
# load text from settings file:
try:
    WANDB_DIR = os.environ['WANDB_DIR']
except KeyError:
    WANDB_DIR = None


def try_get_sweep_id(sweep_cfg_fn, offline=False):
    with open(sweep_cfg_fn, "r") as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)
    if offline:
        print("using a local controller type")
        sweep_config["controller"] = {'type': 'local'}
    current_cfg_hash = hashlib.sha256(json.dumps(sweep_config, sort_keys=True).encode()).hexdigest()
    # see if the sweep for the config already exists
    if os.path.exists(SWEEP_ID_FN):
        with open(SWEEP_ID_FN, "r") as f:
            saved_sweep_hash = f.readline()[:-1]
            saved_sweep_id = f.readline()
        if current_cfg_hash == saved_sweep_hash:
            print('continuing the existing sweep')
            return saved_sweep_id
    # otherwise create an updated sweep
    if os.path.exists(SEMAPHORE_FN):
        time.sleep(2)
        print("config already being created")
    else:
        try:
            with open(SEMAPHORE_FN, "w") as f:
                f.write("creating config")
            sweep_id = wandb.sweep(sweep_config)
            with open(SWEEP_ID_FN, "w") as f:
                f.writelines([str(current_cfg_hash), '\n', sweep_id])
            print("sweep started")
        except Exception as e:
            print(e)
            print_tb(e.__traceback__)
        finally:
            os.remove(SEMAPHORE_FN)
    with open(SWEEP_ID_FN, "r") as f:
        f.readline()
        sweep_id = f.readline()
    return sweep_id


def main(algo=None, env_id=None, group=None, ft_params=None, log_dir='tf_logs', device='cpu'):
    env = gymnasium.make(env_id)
    total_timesteps = env_to_steps[env_id]
    runs_per_hparam = 3
    avg_auc = 0

    # sample the hyperparameters from wandb locally
    wandb_kwargs = {}
    with open(f'hparams/{env_id}/{algo}.yaml') as f:
        default_params = yaml.load(f, Loader=yaml.FullLoader)

    _unique_id = wandb.util.generate_id()

    for i in range(runs_per_hparam):
        unique_id = _unique_id + f"{i}"
        with wandb.init(sync_tensorboard=True,
                        id=unique_id,
                        group=group,
                        dir=WANDB_DIR,
                        **wandb_kwargs) as run:
            print(run.id)
            cfg = wandb.config
            config = cfg.as_dict()
            # save the first config if sampled from wandb to use in the following runs_per_hparam
            wandb_kwargs['config'] = config

            full_config = {}
            full_config.update(default_params)
            if ft_params is not None:
                # Overwrite the default params:
                full_config.update(ft_params)
                # Log the new params (for group / filtering):
                wandb.log(ft_params)
            else:
                full_config.update(config)

            wandb.log({'env_id': env_id})

            agent = UAgent(env, **full_config,
                           device=device, log_interval=env_to_logfreq[env_id],
                           tensorboard_log=log_dir,
                           render=False, )

            # Measure the time it takes to learn:
            agent.learn(total_timesteps=total_timesteps)
            avg_auc += agent.eval_auc
            wandb.log({'avg_auc': avg_auc / runs_per_hparam})
            del agent

    wandb.finish()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--count', type=int, default=10)
    args.add_argument('--project', type=str, default='u-chi-learning')
    args.add_argument('--env_id', type=str, default='LunarLander-v2')
    args.add_argument('--algo', type=str, default='logu')
    args.add_argument('--device', type=str, default='cpu')
    args.add_argument('--offline', action='store_true')
    args.add_argument('--exp-name', type=str, default='EVAL-mean')

    args = args.parse_args()
    # Run a hyperparameter sweep with w&b:
    print("Running a sweep on wandb...")

    sweep_id = try_get_sweep_id(f'sweeps/{args.exp_name}.yaml', offline=args.offline)

    wandb.agent(
        sweep_id,
        function=main(algo=args.algo, env_id=args.env_id, device=args.device, group=args.exp_name),
        count=args.count,
        project=args.project
    )

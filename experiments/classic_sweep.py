import gymnasium
import wandb
import argparse
import yaml
import sys
import os

sys.path.append('darer')
from UAgent import UAgent
from utils import sample_wandb_hyperparams


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

int_hparams = {'train_freq', 'gradient_steps'}

env_id = 'Acrobot-v1'
# load text from settings file:
try:
    WANDB_DIR = os.environ['WANDB_DIR']
except KeyError:
    WANDB_DIR = None


def main(sweep_config=None, algo=None, project=None, ft_params=None, log_dir='tf_logs', device='cpu'):
    def fn():
        env = gymnasium.make(env_id)
        total_timesteps = env_to_steps[env_id]
        runs_per_hparam = 3
        avg_auc = 0

        # sample the hyperparameters from wandb locally
        wandb_kwargs = {"project": project}
        if sweep_config:
            sweep_config["controller"] = {'type': 'local'}
            sampled_params = sample_wandb_hyperparams(sweep_config["parameters"], int_hparams=int_hparams)
            print(f"locally sampled params: {sampled_params}")
            wandb_kwargs['config'] = sampled_params

        with open(f'hparams/{env_id}/{algo}.yaml') as f:
            default_params = yaml.load(f, Loader=yaml.FullLoader)

        with wandb.init(sync_tensorboard=True,
                        dir=WANDB_DIR,
                        **wandb_kwargs) as run:
            for i in range(runs_per_hparam):
                cfg = run.config
                print(run.id)
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

                # cast sampled params to int if they are in int_hparams
                for k in int_hparams:
                    full_config[k] = int(full_config[k])

                agent = UAgent(env, **full_config,
                               device=device, log_interval=env_to_logfreq[env_id],
                               tensorboard_log=log_dir,
                               render=False, )

                # Measure the time it takes to learn:
                agent.learn(total_timesteps=total_timesteps)
                avg_auc += agent.eval_auc
                del agent
            wandb.log({'avg_auc': avg_auc / runs_per_hparam})

        wandb.finish()
    return fn


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--count', type=int, default=10)
    args.add_argument('--project', type=str, default='u-chi-learning')
    args.add_argument('--env-id', type=str, default='LunarLander-v2')
    args.add_argument('--algo', type=str, default='logu')
    args.add_argument('--device', type=str, default='cpu')
    args.add_argument('--exp-name', type=str, default='EVAL-mean')
    args.add_argument("--n-hparam-runs", type=int, default=3, help="number of times to re-train with a single set of hyperparameters")
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
    sweep_cfg = yaml.safe_load(open(f'sweeps/{experiment_name}.yaml'))
    wandb.agent(
        args.sweep,
        project=args.proj,
        count=args.n_runs,
        function=main(sweep_config=sweep_cfg, algo=args.algo, project=args.proj, ft_params=None, device=device))

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

algo_to_agent = {
    'u': UAgent,
    'arSAC': arSAC,
    'logu': LogUAgent
}

int_hparams = {'train_freq', 'gradient_steps'}

# load text from settings file:
try:
    WANDB_DIR = os.environ['WANDB_DIR']
except KeyError:
    WANDB_DIR = None
    
def main(sweep_config=None, env_id=None, algo=None, project=None, ft_params=None, log_dir='tf_logs', device='cpu'):
    # env = gymnasium.make(env_id)
    total_timesteps = env_to_steps.get(env_id, 100_000)
    runs_per_hparam = 3
    avg_auc = 0
    unique_id = wandb.util.generate_id()

    # sample the hyperparameters from wandb locally
    wandb_kwargs = {"project": project}
    if sweep_config:
        sweep_config["controller"] = {'type': 'local'}
        sampled_params = sample_wandb_hyperparams(sweep_config["parameters"], int_hparams=int_hparams)
        print(f"locally sampled params: {sampled_params}")
        wandb_kwargs['config'] = sampled_params

    # Use SQL as default params for U learning 
    with open(f'hparams/{env_id}/{algo}.yaml') as f:
        default_params = yaml.load(f, Loader=yaml.FullLoader)

    for i in range(runs_per_hparam):
        unique_id = unique_id[:-1] + f"{i}"
        with wandb.init(sync_tensorboard=True, 
                        id=unique_id,
                        dir=WANDB_DIR,
                        **wandb_kwargs) as run: 
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

            # Choose the algo appropriately
            Agent = algo_to_agent.get(algo, UAgent)

            agent = Agent(env_id, **full_config,
                                device=device, log_interval=env_to_logfreq.get(env_id, 500),
                                tensorboard_log=log_dir,
                                render=False,)

            # Measure the time it takes to learn:
            agent.learn(total_timesteps=total_timesteps)
            avg_auc += agent.eval_auc
            wandb.log({'avg_auc': avg_auc / runs_per_hparam})
            del agent

    wandb.finish()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--count', type=int, default=10)
    args.add_argument('--project', type=str, default='eval-full-ft')
    args.add_argument('--env_id', type=str, default='CartPole-v1')
    args.add_argument('--algo', type=str, default='logu')
    args.add_argument('--device', type=str, default='cpu')
    args.add_argument('--exp-name', type=str, default='EVAL')

    args = args.parse_args()

    # Run a hyperparameter sweep with w&b:
    print("Running a sweep on wandb...")
    sweep_cfg = yaml.safe_load(open(f'sweeps/{args.exp_name}.yaml'))

    for i in range(args.count):
        main(sweep_cfg, env_id=args.env_id, algo=args.algo, project=args.project, device=args.device)


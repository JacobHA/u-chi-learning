import sys
sys.path.append("darer")
import argparse
import wandb
from SoftQAgent import SoftQAgent
from LogUAgent import LogUAgent
from UAgent import UAgent
from hparams import id_to_hparam_dicts


env_id_to_timesteps = {
    'CartPole-v1': 50_000,
    'Acrobot-v1': 50_000,
    'LunarLander-v2': 500_000,
    'PongNoFrameskip-v4': 1_000_000,
    'MountainCar-v0': 100_000,
}

int_hparams = ['batch_size',
               'target_update_interval',
               'theta_update_interval',
               'gradient_steps',
               'train_freq',
               'learning_starts',
               'buffer_size']


def runner(config=None, run=None):
    # Convert the necessary kwargs to ints:
    hconfig = id_to_hparam_dicts[env_id][algo]
    total_timesteps = env_id_to_timesteps[env_id]


    hconfig['buffer_size'] = total_timesteps

    LOG_INTERVAL = 500
    device = 'cpu'
    runs_per_hparam = 3
    auc = 0

    for _ in range(runs_per_hparam):
        wandb.log({'env_id': env_id})
        wandb.log({'beta': hconfig['beta']})
        # config.pop('device')
        # config.pop('env_id')
        # config['buffer_size'] = 300_000
        agent = UAgent(env_id, **config, log_interval=LOG_INTERVAL, use_wandb=True,
                       **hconfig,
                        render=False,
                        device=device)
        agent.learn(total_timesteps=total_timesteps,)

        auc += agent.eval_auc
    auc /= runs_per_hparam
    wandb.log({'avg_eval_auc': auc})


def wandb_agent():
    with wandb.init(sync_tensorboard=False, monitor_gym=False, dir='logs') as run:
        cfg = run.config
        dict_cfg = cfg.as_dict()
        runner(dict_cfg, run=run)


if __name__ == "__main__":
    # Parse the "algo" argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default='cpu')
    parser.add_argument("-c", "--count", type=int, default=15_000)
    parser.add_argument("-e", "--entity", type=str, default='jacobhadamczyk')
    parser.add_argument("-p", "--project", type=str,
                        default='u-chi-learning-darer')
    parser.add_argument("-a", "--algo", type=str, default='sql')
    parser.add_argument("-env", "--env_id", type=str, default='Acrobot-v1')
    args = parser.parse_args()
    entity = args.entity
    project = args.project
    algo = args.algo

    sweep_id = 'mhovfqyc'
    env_id = args.env_id
    device = args.device

    full_sweep_id = f"{entity}/{project}/{sweep_id}"

    # TODO: Before calling the agent on this full_sweep_id, make sure it exists (i.e. the project and sweep):
    # test_sweep_existence(full_sweep_id)
    wandb.agent(full_sweep_id, function=wandb_agent, count=args.count)

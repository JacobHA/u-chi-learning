import sys
sys.path.append("darer")
import argparse
import wandb
from SoftQAgent import SoftQAgent
from LogUAgent import LogUAgent
from UAgent import UAgent


env_id = 'CartPole-v1'
# env_id = 'MountainCar-v0'
# env_id = 'LunarLander-v2'
# env_id = 'Pong-v4'
# env_id = 'HalfCheetah-v4'
# env_id = 'Acrobot-v1'
# env_id = 'Pendulum-v1'
env_id = None

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

    if 'buffer_size' not in config:
        config['buffer_size'] = total_timesteps

    LOG_INTERVAL = 500
    device = 'cpu'
    runs_per_hparam = 3

    for _ in range(runs_per_hparam):
        wandb.log({'env_id': env_id})
        if algo == 'u':
            # config.pop('device')
            # config.pop('env_id')
            # config['buffer_size'] = 300_000
            agent = UAgent(env_id, **config, log_interval=LOG_INTERVAL, use_wandb=True,
                           render=False,
                           use_rawlik=False,
                           device=device)

        elif algo == 'sql':
            agent = SoftQAgent(env_id, **config, log_interval=LOG_INTERVAL, use_wandb=True,
                               render=False,
                               device=device)
        wandb.log({'agent_name': agent.algo_name})

        early_stopped = agent.learn(total_timesteps=total_timesteps,)
        # early_stop=env_to_early_stop_dict[env_id])
        if early_stopped:
            break
        auc += agent.eval_auc
    auc /= runs_per_hparam
    wandb.log({'avg_eval_auc': auc})


def wandb_agent():
    with wandb.init(sync_tensorboard=False, monitor_gym=False, dir='logs') as run:
        cfg = run.config
        dict_cfg = cfg.as_dict()
        runner(dict_cfg, run=run)


if __name__ == "__main__":
    algo_to_sweep_id = {'u': '27ojzk9g',  # '6301y2oc',
                        'sql': 'frb1998p'}
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

    sweep_id = algo_to_sweep_id[algo]
    if 'NoFrameskip' in args.env_id:
        # sweep_id = '5gwi5rfx'
        sweep_id = 'e6nnzdsf'
    if args.env_id == 'LunarLander-v2':
        sweep_id = 'y6gv3ss2'
    env_id = args.env_id
    device = args.device

    full_sweep_id = f"{entity}/{project}/{sweep_id}"

    # TODO: Before calling the agent on this full_sweep_id, make sure it exists (i.e. the project and sweep):
    # test_sweep_existence(full_sweep_id)
    wandb.agent(full_sweep_id, function=wandb_agent, count=args.count)

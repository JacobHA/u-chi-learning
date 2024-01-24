import sys
sys.path.append("darer")
from UAgent import UAgent
from LogUAgent import LogUAgent
from SoftQAgent import SoftQAgent
import wandb
import argparse

LOG_INTERVAL = 500

env_id = 'CartPole-v1'
# env_id = 'MountainCar-v0'
# env_id = 'LunarLander-v2'
# env_id = 'Pong-v4'
# env_id = 'HalfCheetah-v4'
# env_id = 'Acrobot-v1'
# env_id = 'Pendulum-v1'
env_id = None

env_to_early_stop_dict = {
    'CartPole-v1': {'reward': 200, 'steps': 10_000},
    'Acrobot-v1': {'reward': -200, 'steps': 10_000},
    'MountainCar-v0': {'reward': -190, 'steps': 20_000},
    'LunarLander-v2': {'reward': 0, 'steps': 20_000},
    'PongNoFrameskip-v4': {'reward': -22, 'steps': 20_000},
}

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
    for int_kwarg in int_hparams:
        try:
            config[int_kwarg] = int(config[int_kwarg])
        except KeyError:
            pass  # use default value

    # beta_end = config.pop('final_beta_multiplier') * config['beta']
    config['gradient_steps'] = config['train_freq']
    runs_per_hparam = 3
    auc = 0
    total_timesteps = env_id_to_timesteps[env_id]

    # check if learning starts ratio is a config key:
    if 'learning_starts_ratio' in config:
        learn_ratio = config.pop('learning_starts_ratio')
        config['learning_starts'] = total_timesteps * learn_ratio

    else:
        pass

    if 'buffer_size' not in config:
        config['buffer_size'] = total_timesteps

    for _ in range(runs_per_hparam):
        wandb.log({'env_id': env_id})
        if algo == 'u':

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
    algo_to_sweep_id = {'u': '6301y2oc',
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
    env_id = args.env_id
    device = args.device
    # from disc_envs import get_environment
    # env_id = get_environment('Pendulum21', nbins=3, max_episode_steps=200, reward_offset=0)

    full_sweep_id = f"{entity}/{project}/{sweep_id}"

    # TODO: Before calling the agent on this full_sweep_id, make sure it exists (i.e. the project and sweep):
    # test_sweep_existence(full_sweep_id)
    wandb.agent(full_sweep_id, function=wandb_agent, count=args.count)

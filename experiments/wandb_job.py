from new_logac import LogUActor
from MultiLogU import LogULearner
import wandb
import argparse
import sys
sys.path.append("darer")

# env_id = 'CartPole-v1'
# env_id = 'MountainCar-v0'
env_id = 'LunarLander-v2'
# env_id = 'HalfCheetah-v4'
# env_id = 'Acrobot-v1'
# env_id = 'Pendulum-v1'
env_id = None


def runner(config=None, run=None, device='cpu'):
    # Convert the necessary kwargs to ints:
    for int_kwarg in ['batch_size', 'target_update_interval', 'theta_update_interval', 'gradient_steps',
                      'train_freq', 'learning_starts', 'buffer_size']:
        config[int_kwarg] = int(config[int_kwarg])
    # config['buffer_size'] = 10_000
    # config['gradient_steps'] = 1
    # config['train_freq'] = 1
    # config['learning_starts'] = 1_000
    # config.pop('actor_learning_rate')
    beta_schedule = config.pop('beta_scheduler')
    beta_end = config.pop('final_beta_multiplier') * config['beta']
    runs_per_hparam = 3
    auc = 0
    wandb.log({'env_id': env_id})

    for _ in range(runs_per_hparam):
        model = LogULearner(env_id, **config, log_interval=50, use_wandb=True,
                            device=device, render=0, beta_end=beta_end)
        model.learn(total_timesteps=25_000, beta_schedule=beta_schedule)
        auc += model.eval_auc
    auc /= runs_per_hparam
    wandb.log({'avg_eval_auc': auc})


def wandb_agent():
    with wandb.init(sync_tensorboard=False, monitor_gym=False, dir='logs') as run:
        cfg = run.config
        dict_cfg = cfg.as_dict()
        runner(dict_cfg, run=run, device=args.device)


if __name__ == "__main__":
    # set up wandb variables (TODO: these should be set up as globals per user):
    # Parse the "algo" argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default='cpu')
    parser.add_argument("-c", "--count", type=int, default=100)
    parser.add_argument("-e", "--entity", type=str, default='jacobhadamczyk')
    parser.add_argument("-p", "--project", type=str, default='LogU-Cartpole')
    parser.add_argument("-s", "--sweep_id", type=str, default='xcmjwq1t')
    parser.add_argument("-env", "--env_id", type=str, default='ALE/Pong-v5')
    args = parser.parse_args()
    entity = args.entity
    project = args.project
    sweep_id = args.sweep_id
    env_id = args.env_id

    full_sweep_id = f"{entity}/{project}/{sweep_id}"

    # TODO: Before calling the agent on this full_sweep_id, make sure it exists (i.e. the project and sweep):
    # test_sweep_existence(full_sweep_id)
    wandb.agent(full_sweep_id, function=wandb_agent, count=args.count)
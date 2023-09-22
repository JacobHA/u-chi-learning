import argparse
import wandb
import gym
from LogU import LogULearner
from hparams import cartpole_hparams

env = gym.make('CartPole-v1')
config = cartpole_hparams

def runner(config=config):
    config['hidden_dim'] = args.hidden_dim
    model = LogULearner(env, **config, log_dir='ft/cartpole', log_interval=100)
    model.learn_online(total_timesteps=50_000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, default=256)
    parser.add_argument("-c", "--count", type=int, default=1)
    
    args = parser.parse_args()

    for i in range(args.count):
        runner()
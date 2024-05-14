import gymnasium as gym
import sys
import argparse
sys.path.append('darer')

from UAgent import UAgent
from ASQL import ASQL
from ASAC import ASAC
from SoftQAgent import SoftQAgent
from CustomSAC import CustomSAC
from arDDPG import arDDPG

from gymnasium.wrappers import RecordVideo

def main(path=None, video=False, n_steps=1000, n_episodes=1):
    if '/' in path:
        path_strs = path.split('/')
        model_name = path_strs[-1]
    else:
        model_name = path
    model_cls_name, env_id = model_name.split('_')
    model_cls = globals()[model_cls_name]
    # env = gym.make(env_id)
    model = model_cls.load(path, render=True, save_best=False, render_mode='rgb_array' if video else 'human')
    # model.eval_env.render_mode = 'rgb_array'
    if video:
        model.eval_env = RecordVideo(
            model.eval_env,
            video_folder='best_models/',
            episode_trigger=lambda episode_id: episode_id % 1 == 0,
            video_length=n_steps
        )
    model.evaluate(n_episodes, render=True)
    model.eval_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='best_models/ASQL_PongNoFrameskip-v4', type=str)
    parser.add_argument('--video', default=False, type=bool)
    parser.add_argument('--n_steps', default=10000, type=int)
    parser.add_argument('--n_episodes', default=10, type=int)
    args = parser.parse_args()
    main(**vars(args))
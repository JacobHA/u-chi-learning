import gymnasium as gym
import sys
import argparse
sys.path.append('darer')

from ASQL import ASQL
from ASAC import ASAC
from SoftQAgent import SoftQAgent
from CustomSAC import CustomSAC
from arDDPG import arDDPG

from gymnasium.wrappers import RecordVideo

def main(path):
    if '/' in path:
        path_strs = path.split('/')
        model_name = path_strs[-1]
    else:
        model_name = path
    model_cls_name, env_id = model_name.split('_')
    model_cls = globals()[model_cls_name]
    # env = gym.make(env_id)
    model = model_cls.load(path, render=True, save_best=False, render_mode='rgb_array')
    # model.eval_env.render_mode = 'rgb_array'
    model.eval_env = RecordVideo(
        model.eval_env,
        video_folder='best_models/',
        episode_trigger=lambda episode_id: episode_id % 1 == 0,
        video_length=10000
    )
    model.evaluate(10, render=True)
    model.eval_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='best_models/ASQL_PongNoFrameskip-v4', type=str)
    args = parser.parse_args()
    main(args.path)
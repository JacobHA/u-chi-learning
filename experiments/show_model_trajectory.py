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


def main(path):
    if '/' in path:
        path_strs = path.split('/')
        model_name = path_strs[-1]
    else:
        model_name = path
    model_cls_name, env_id = model_name.split('_')
    model_cls = globals()[model_cls_name]
    # env = gym.make(env_id)
    model = model_cls.load(path, render=True)
    model.evaluate(1, render=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='best_models/ASAC_HalfCheetah-v4', type=str)
    args = parser.parse_args()
    main(args.path)
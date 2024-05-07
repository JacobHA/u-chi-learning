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
    path_strs = path.split('/')
    model_name = path_strs[-1]
    model_cls_name, env_id = model_name.split('_')
    model_cls = globals()[model_cls_name]
    env = gym.make(env_id, render=True)
    model = model_cls.load(path)
    model.set_env(env)
    model.evaluate(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('best_models/', type=str)
    args = parser.parse_args()
    main(args.path)
import os
import shutil

import pytest
import sys
sys.path.append('darer')
sys.path.append('../darer')

from UAgent import UAgent
from ASQL import ASQL
from ASAC import ASAC
from SoftQAgent import SoftQAgent
from CustomSAC import CustomSAC
from arDDPG import arDDPG


def test_saving_and_loading():
    agents = [UAgent, ASQL, ASAC, SoftQAgent, CustomSAC, arDDPG]
    act_cont = [False, False, True, False, True, True]
    # continuous and discrete envs for corresponding agents
    for i, agent in enumerate(agents):
        env_id = 'MountainCarContinuous-v0' if act_cont[i] else 'CartPole-v1'
        agent = agent(env_id=env_id, tensorboard_log='test')
        agent.save(f"test/test-{str(agent)}")
        agent.load(f"test/test-{str(agent)}")
        assert isinstance(agent, agent.__class__)
    shutil.rmtree('test')


if __name__ == '__main__':
    test_saving_and_loading()
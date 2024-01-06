import pytest
import sys
import numpy as np
sys.path.append('darer')
from darer.LogUAgent import LogU
import gymnasium as gym
import torch

seed = 42
env_id = 'CartPole-v1'
env = gym.make(env_id)

@pytest.fixture
def logu_agent():
    return LogU(env_id=env_id, learning_starts=0)  # Initialize LogU with a specific environment ID

def test_logu_agent_creation(logu_agent):
    assert isinstance(logu_agent, LogU)

def test_exploration_policy(logu_agent):
    state = env.observation_space.sample()
    action = logu_agent.exploration_policy(state)
    assert action in env.action_space  # Assuming the action is valid

def test_evaluation_policy(logu_agent):
    state = env.observation_space.sample()
    action = logu_agent.evaluation_policy(state)
    assert action in env.action_space  # Assuming the action is valid

def test_rollout(logu_agent):
    logu_agent.learn(total_timesteps=10)
    assert logu_agent.env_steps == 10

def test_evaluate(logu_agent):
    eval_rwd = logu_agent.evaluate()
    assert isinstance(eval_rwd, float)
    rmin, rmax = env.reward_range
    assert rmin <= eval_rwd <= rmax

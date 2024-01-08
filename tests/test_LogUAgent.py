import pytest
import sys
import numpy as np
sys.path.append('darer')
from darer.LogUAgent import LogU
import gymnasium as gym
from pytest_mock import mocker  # Import the mocker fixture

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
    action, kl = logu_agent.exploration_policy(state)
    assert action in env.action_space  # Assuming the action is valid
    assert kl >= 0

def test_evaluation_policy(logu_agent):
    state = env.observation_space.sample()
    action = logu_agent.evaluation_policy(state)
    assert action in env.action_space  # Assuming the action is valid

def test_rollout(logu_agent):
    logu_agent.learn(total_timesteps=10)
    assert logu_agent.env_steps == 10

def test_on_step_increment_environment_steps(logu_agent):
    initial_env_steps = logu_agent.env_steps
    logu_agent.learn(total_timesteps=1)
    assert logu_agent.env_steps == initial_env_steps + 1


def test_evaluate(logu_agent):
    eval_rwd = logu_agent.evaluate()
    assert isinstance(eval_rwd, float)
    rmin, rmax = env.reward_range
    assert rmin <= eval_rwd <= rmax

def test_learn_calls_train_descent_if_learning_starts_exceeded(logu_agent, mocker):
    mocker.patch.object(logu_agent, '_train')
    logu_agent.learning_starts = 5
    logu_agent.train_freq = 1
    logu_agent.gradient_steps = 1
    logu_agent.learn(6)
    logu_agent._train.assert_called_once()

def test_learn_calls_gradient_descent_if_learning_starts_exceeded(logu_agent, mocker):
    mocker.patch.object(logu_agent, 'gradient_descent')
    train_freq = 2
    gradient_steps = 6
    learning_starts = 5
    learn_steps = 7
    logu_agent.learning_starts = learning_starts
    logu_agent.train_freq = train_freq
    logu_agent.gradient_steps = gradient_steps
    logu_agent.learn(learn_steps)
    num_calls = (learn_steps - learning_starts) // train_freq * gradient_steps
    assert logu_agent.gradient_descent.call_count == num_calls

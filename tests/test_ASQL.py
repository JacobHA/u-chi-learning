import pytest
import sys
sys.path.append('darer')
sys.path.append('../darer')
from ASQL import ASQL
import gymnasium as gym
from pytest_mock import mocker  # Import the mocker fixture

seed = 42
env_id = 'CartPole-v1'
env = gym.make(env_id)

@pytest.fixture
def asql_agent():
    return ASQL(env_id=env_id, learning_starts=0)  # Initialize LogU with a specific environment ID


def test_asql_agent_creation(asql_agent):
    assert isinstance(asql_agent, ASQL)


def test_exploration_policy(asql_agent):
    state = env.observation_space.sample()
    action, kl = asql_agent.exploration_policy(state)
    assert action in env.action_space  # Assuming the action is valid
    assert kl >= 0


def test_evaluation_policy(asql_agent):
    state = env.observation_space.sample()
    action = asql_agent.evaluation_policy(state)
    assert action in env.action_space  # Assuming the action is valid


def test_rollout(asql_agent):
    asql_agent.learn(total_timesteps=10)
    assert asql_agent.env_steps == 10


def test_on_step_increment_environment_steps(asql_agent):
    initial_env_steps = asql_agent.env_steps
    asql_agent.learn(total_timesteps=1)
    assert asql_agent.env_steps == initial_env_steps + 1


def test_evaluate(asql_agent):
    eval_rwd = asql_agent.evaluate()
    assert isinstance(eval_rwd, float)
    rmin, rmax = env.reward_range
    assert rmin <= eval_rwd <= rmax

def test_learn_calls_train_descent_if_learning_starts_exceeded(asql_agent, mocker):
    mocker.patch.object(asql_agent, '_train')
    asql_agent.learning_starts = 5
    asql_agent.train_freq = 1
    asql_agent.gradient_steps = 1
    asql_agent.learn(6)
    asql_agent._train.assert_called_once()


def test_learn_calls_gradient_descent_if_learning_starts_exceeded(asql_agent, mocker):
    mocker.patch.object(asql_agent, 'gradient_descent')
    train_freq = 2
    gradient_steps = 6
    learning_starts = 5
    learn_steps = 7
    asql_agent.learning_starts = learning_starts
    asql_agent.train_freq = train_freq
    asql_agent.gradient_steps = gradient_steps
    asql_agent.learn(learn_steps)
    num_calls = (learn_steps - learning_starts) // train_freq * gradient_steps
    assert asql_agent.gradient_descent.call_count == num_calls


import pytest
import sys
sys.path.append('darer')
# Import your BaseAgent class from your code
from darer.BaseAgent import BaseAgent


@pytest.fixture
def base_agent():
    # Initialize a BaseAgent instance with minimal configuration for testing
    return BaseAgent(env_id='CartPole-v1')


def test_initialize_networks_raises_not_implemented(base_agent):
    # Ensure _initialize_networks raises NotImplementedError
    with pytest.raises(NotImplementedError):
        base_agent._initialize_networks()


def test_exploration_policy_raises_not_implemented(base_agent):
    # Ensure exploration_policy raises NotImplementedError
    with pytest.raises(NotImplementedError):
        state = None  # Provide a sample state for testing
        base_agent.exploration_policy(state)


def test_gradient_descent_raises_not_implemented(base_agent):
    # Ensure gradient_descent raises NotImplementedError
    with pytest.raises(NotImplementedError):
        batch = None  # Provide a sample batch for testing
        grad_step = 0  # Provide a sample gradient step for testing
        base_agent.gradient_descent(batch, grad_step)


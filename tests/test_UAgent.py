# import pytest
# import sys
# sys.path.append('darer')
# sys.path.append('../darer')
# from UAgent import UAgent
# import gymnasium as gym
# from pytest_mock import mocker  # Import the mocker fixture

# seed = 42
# env_id = 'CartPole-v1'
# env = gym.make(env_id)

# @pytest.fixture
# def u_agent():
#     return UAgent(env_id=env_id, learning_starts=0)  # Initialize u with a specific environment ID


# def test_u_agent_creation(u_agent):
#     assert isinstance(u_agent, UAgent)


# def test_exploration_policy(u_agent):
#     state = env.observation_space.sample()
#     action, kl = u_agent.exploration_policy(state)
#     assert action in env.action_space  # Assuming the action is valid
#     assert kl >= 0


# def test_evaluation_policy(u_agent):
#     state = env.observation_space.sample()
#     action = u_agent.evaluation_policy(state)
#     assert action in env.action_space  # Assuming the action is valid


# def test_rollout(u_agent):
#     u_agent.learn(total_timesteps=10)
#     assert u_agent.env_steps == 10


# def test_on_step_increment_environment_steps(u_agent):
#     initial_env_steps = u_agent.env_steps
#     u_agent.learn(total_timesteps=1)
#     assert u_agent.env_steps == initial_env_steps + 1


# def test_evaluate(u_agent):
#     eval_rwd = u_agent.evaluate()
#     assert isinstance(eval_rwd, float)
#     rmin, rmax = env.reward_range
#     assert rmin <= eval_rwd <= rmax

# def test_learn_calls_train_descent_if_learning_starts_exceeded(u_agent, mocker):
#     mocker.patch.object(u_agent, '_train')
#     u_agent.learning_starts = 5
#     u_agent.train_freq = 1
#     u_agent.gradient_steps = 1
#     u_agent.learn(6)
#     u_agent._train.assert_called_once()


# def test_learn_calls_gradient_descent_if_learning_starts_exceeded(u_agent, mocker):
#     mocker.patch.object(u_agent, 'gradient_descent')
#     train_freq = 2
#     gradient_steps = 6
#     learning_starts = 5
#     learn_steps = 7
#     u_agent.learning_starts = learning_starts
#     u_agent.train_freq = train_freq
#     u_agent.gradient_steps = gradient_steps
#     u_agent.learn(learn_steps)
#     num_calls = (learn_steps - learning_starts) // train_freq * gradient_steps
#     assert u_agent.gradient_descent.call_count == num_calls
#     assert u_agent._n_updates == num_calls

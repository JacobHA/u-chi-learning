import torch
import numpy as np
import pytest
import gymnasium as gym
from Models import OnlineLogUNets, LogUNet

num_actions = 5
num_nets = 3

class DummyEnv:
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(10,))
        self.action_space = gym.spaces.Discrete(num_actions)

@pytest.fixture
def online_nets():
    # Create a list of nets for testing
    dummy_env = DummyEnv()
    # Set the torch random seed:
    torch.manual_seed(0)
    list_of_nets = [LogUNet(dummy_env, device='cpu', hidden_dim=8) for _ in range(num_nets)]
    return OnlineLogUNets(list_of_nets, torch.min)

def test_greedy_action(online_nets):
    state = DummyEnv().observation_space.sample()
    action = online_nets.choose_action(state, greedy=True)
    assert action in DummyEnv().action_space

def test_choose_action(online_nets):
    state = DummyEnv().observation_space.sample()
    action = online_nets.choose_action(state)
    assert action in DummyEnv().action_space

def test_parameters(online_nets):
    # Test the parameters method

    # Call the parameters method
    parameters = online_nets.parameters()

    # Check if the returned parameters match the parameters of the nets
    for net, net_parameters in zip(online_nets, parameters):
        for param, net_param in zip(net.parameters(), net_parameters):
            assert torch.all(torch.eq(param, net_param))

def test_clip_grad_norm(online_nets):
    # Test the clip_grad_norm method

    for _ in range(5):
        # Calculate losses based on online_nets values and distance to 1:
        losses = [torch.abs(net(100*torch.ones(10)) - torch.ones(num_actions)).mean() for net in online_nets]
        total_loss = sum(losses)
        total_loss.backward()
    # Want to ensure clipping is always performed so set eps to a very small value
    eps = 1e-10

    # First ensure the gradients will be clipped
    for net in online_nets:
        for param in net.parameters():
            assert param.grad.norm() > eps

    # Call the clip_grad_norm method
    online_nets.clip_grad_norm(eps)

    # Check if the gradients are clipped
    for net in online_nets:
        for param in net.parameters():
            assert torch.all(param.grad.norm() <= eps)

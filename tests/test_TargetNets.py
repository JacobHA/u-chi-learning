import copy
import pytest
import torch
from Models import TargetNets

@pytest.fixture
def target_nets():
    # Create a list of nets for testing
    num_nets = 3
    list_of_nets = [torch.nn.Linear(10, 5) for _ in range(num_nets)]
    return TargetNets(list_of_nets)

@pytest.fixture
def online_nets():
    # Create a list of nets for polyak testing
    num_nets = 3
    list_of_nets = [torch.nn.Linear(10, 5) for _ in range(num_nets)]
    return TargetNets(list_of_nets)

def test_load_state_dict(target_nets):
    # Create some dummy state_dicts
    original_state_dicts = [net.state_dict() for net in target_nets]

    # Modify the state dicts (for testing purposes)
    modified_state_dicts = []
    for state_dict in original_state_dicts:
        modified_state_dict = {key: value * 2 for key, value in state_dict.items()}
        modified_state_dicts.append(modified_state_dict)

    # Load the modified state dicts into the TargetNets
    target_nets.load_state_dicts(modified_state_dicts)

    # Check if the nets have been updated
    for net, modified_state_dict in zip(target_nets, modified_state_dicts):
        for param_name, param in net.named_parameters():
            # Ensure each parameter is updated correctly
            assert torch.all(torch.eq(param, modified_state_dict[param_name]))

import copy

def test_polyak(target_nets, online_nets):
    # Create some dummy parameters

    # Perform polyak update
    tau = 0.25
    with torch.no_grad():
        # deep copy the original nets:
        original_nets = copy.deepcopy(target_nets)
        # SB3 Polyak update:
        target_nets.polyak(online_nets, tau)

        # Check if the nets have been updated
        for updated_net, mixture_net, original_net in zip(target_nets, online_nets, original_nets):
            for updated_param, mix_param, original_param in zip(updated_net.parameters(), mixture_net.parameters(), original_net.parameters()):
                # Ensure each parameter is updated correctly
                mix_param.data.mul_(tau)
                original_param.data.mul_(1 - tau)
                result = torch.add(mix_param.data, original_param.data)
                assert torch.allclose(updated_param.data, result)

def test_zero_polyak(target_nets, online_nets):
    # Perform polyak update
    tau = 0.0
    with torch.no_grad():
        # deep copy the original nets:
        original_nets = copy.deepcopy(target_nets)
        target_nets.polyak(online_nets, tau)

        # Check the target nets params haven't changed:
        for original_target, updated_target in zip(original_nets, target_nets):
            for original_param, updated_param in zip(original_target.parameters(), updated_target.parameters()):
                assert torch.allclose(original_param, updated_param)
        
def test_one_polyak(target_nets, online_nets):
    # Perform polyak update
    tau = 1.0
    with torch.no_grad():
        target_nets.polyak(online_nets, tau)

        # Check the target nets params haven't changed:
        for online_net, updated_target in zip(online_nets, target_nets):
            for online_param, updated_param in zip(online_net.parameters(), updated_target.parameters()):
                assert torch.equal(online_param, updated_param)


def test_parameters(target_nets):
    # Get parameters using the parameters method
    parameters = target_nets.parameters()

    # Check if the returned parameters match the parameters of the nets
    for net, net_parameters in zip(target_nets, parameters):
        for param, net_param in zip(net.parameters(), net_parameters):
            assert torch.all(torch.eq(param, net_param))

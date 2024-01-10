import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from stable_baselines3.common.utils import zip_strict
from gymnasium import spaces
import gymnasium as gym
from torch.optim.lr_scheduler import StepLR, MultiplicativeLR, LinearLR, ExponentialLR, LRScheduler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from sb3preprocessing import is_image_space, preprocess_obs, get_action_dim, get_flattened_obs_dim
from utils import is_tabular

def is_image_space_simple(observation_space, is_vector_env=False):
    if is_vector_env:
        return isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 4
    return isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3
NORMALIZE_IMG = True


def model_initializer(is_image_space,
                      observation_space,
                      nA,
                      activation,
                      hidden_dim,
                      device):
    model = nn.Sequential()

    # check if image:
    if is_image_space:
        nS = get_flattened_obs_dim(observation_space)
        # Use a CNN:
        n_channels = observation_space.shape[2]
        model.extend(nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4, device=device),
            activation(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, device=device),
            activation(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, device=device),
            nn.Flatten(start_dim=1),
        ))
        # calculate resulting shape for FC layers:
        rand_inp = observation_space.sample()
        x = torch.tensor(rand_inp, device=device, dtype=torch.float32)  # Convert to PyTorch tensor
        x = x.detach()
        x = preprocess_obs(x, observation_space, normalize_images=NORMALIZE_IMG)
        x = x.permute([2,0,1]).unsqueeze(0)
        flat_size = model(x).shape[1]
        # with torch.no_grad():
        #     n_flatten = model(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        print(f"Using a CNN with {flat_size}-dim. outputs.")

        model.extend(nn.Sequential(
            nn.Linear(flat_size, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, nA),
        ))

    else:
        if isinstance(observation_space, spaces.Discrete):
            nS = observation_space.n
            input_dim = nS
        else:    
            nS = observation_space.shape
            input_dim = nS[0]
        
        # Use a simple MLP:
        model.extend(nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=torch.float32),
            activation(),
            nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
            activation(),
            nn.Linear(hidden_dim, nA, dtype=torch.float32),
        ))
        # intialize weights with xavier:
        # for m in model:
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight, gain=1)
        #         nn.init.constant_(m.bias, 0)

    return model, nS

class LogUNet(nn.Module):
    def __init__(self, env, device='cuda', hidden_dim=256, activation=nn.ReLU):
        super(LogUNet, self).__init__()
        self.using_vector_env = isinstance(env.action_space, gym.spaces.MultiDiscrete)
        self.env = env
        if self.using_vector_env:
            self.observation_space = self.env.single_observation_space
            self.action_space = self.env.single_action_space
        else:
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
        self.nA = self.action_space.n
        # do the check on an env before wrapping it
        self.is_image_space = is_image_space_simple(self.env.observation_space, self.using_vector_env)
        self.is_tabular = is_tabular(env)
        self.device = device
        model, nS = model_initializer(self.is_image_space,
                                  self.observation_space,
                                  self.nA,
                                  activation,
                                  hidden_dim,
                                  self.device)

        model.to(self.device)
        self.model = model
        self.nS = nS
        
        self.to(device)
     
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)  # Convert to PyTorch tensor
        
        # x = x.detach()
        x = preprocess_obs(x, self.env.observation_space, normalize_images=NORMALIZE_IMG)
        assert x.dtype == torch.float32, "Input must be a float tensor."

        # Reshape the image:
        if self.is_image_space:
            if len(x.shape) == 3:
                # Single image
                x = x.permute([2,0,1])
                x = x.unsqueeze(0)
            else:
                # Batch of images
                x = x.permute([0,3,1,2])
        elif self.is_tabular:
            # Single state
            if x.shape[0] == self.nS:
                x = x.unsqueeze(0)
            else: 
                x = x.squeeze(1)
                pass
        else:
            if len(x.shape) > len(self.nS):
                # in batch mode:
                pass
            else:
                # is a single state
                if isinstance(x.shape, torch.Size):
                    if x.shape == self.nS:
                        x = x.unsqueeze(0)
                else:
                    if (x.shape == self.nS).all():
                        x = x.unsqueeze(0)

        x = self.model(x)
        return x
        
   

class EmptyScheduler(LRScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def get_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]

str_to_scheduler = {
    "step": (StepLR, {'step_size': 100_000, 'gamma': 0.5}),
    # "MultiplicativeLR": (MultiplicativeLR, ()), 
    "linear": (LinearLR, {"start_factor":1./3, "end_factor":1.0, "last_epoch":-1}), 
    "exponential": (ExponentialLR, {'gamma': 0.9999}), 
    "none": (EmptyScheduler, {"last_epoch":-1})
}

class Optimizers():
    def __init__(self, list_of_optimizers: list, scheduler_str: str = 'none'):
        self.optimizers = list_of_optimizers
        scheduler_str = scheduler_str.lower()
        scheduler, params = str_to_scheduler[scheduler_str]
        
        self.schedulers = [scheduler(opt, **params) for opt in self.optimizers]

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def step(self):
        for opt, scheduler in zip(self.optimizers, self.schedulers):
            opt.step()
            scheduler.step()

    def get_lr(self):
        return self.schedulers[0].get_lr()[0]

class TargetNets():
    def __init__(self, list_of_nets):
        self.nets = list_of_nets
    def __len__(self):
        return len(self.nets)
    def __iter__(self):
        return iter(self.nets)

    def load_state_dicts(self, list_of_state_dicts):
        """
        Load state dictionaries into target networks.

        Args:
            list_of_state_dicts (list): A list of state dictionaries to load into the target networks.

        Raises:
            ValueError: If the number of state dictionaries does not match the number of target networks.
        """
        if len(list_of_state_dicts) != len(self):
            raise ValueError("Number of state dictionaries does not match the number of target networks.")
        
        for online_net_dict, target_net in zip(list_of_state_dicts, self):
            
            target_net.load_state_dict(online_net_dict)

    def polyak(self, online_nets, tau):
        """
        Perform a Polyak (exponential moving average) update for target networks.

        Args:
            online_nets (list): A list of online networks whose parameters will be used for the update.
            tau (float): The update rate, typically between 0 and 1.

        Raises:
            ValueError: If the number of online networks does not match the number of target networks.
        """
        if len(online_nets) != len(self.nets):
            raise ValueError("Number of online networks does not match the number of target networks.")

        with torch.no_grad():
            # zip does not raise an exception if length of parameters does not match.
            for new_params, target_params in zip(online_nets.parameters(), self.parameters()):
                for new_param, target_param in zip_strict(new_params, target_params):
                    target_param.data.mul_(tau).add_(new_param.data, alpha=1.0-tau)

    def parameters(self):
        """
        Get the parameters of all target networks.

        Returns:
            list: A list of network parameters for each target network.
        """
        return [net.parameters() for net in self.nets]


class OnlineNets():
    """
    A utility class for managing online networks in reinforcement learning.

    Args:
        list_of_nets (list): A list of online networks.
    """
    def __init__(self, list_of_nets, aggregator_fn, is_vector_env=False):
        self.nets = list_of_nets
        self.nA = list_of_nets[0].nA
        self.device = list_of_nets[0].device
        self.aggregator_fn = aggregator_fn
        self.is_vector_env = is_vector_env

    def __len__(self):
        return len(self.nets)
    
    def __iter__(self):
        return iter(self.nets)

    def choose_action(self, state, greedy=False, prior=None):
        raise NotImplementedError

    def parameters(self):
        return [net.parameters() for net in self]

    def clip_grad_norm(self, max_grad_norm):
        for net in self:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)


class OnlineLogUNets(OnlineNets):
    def __init__(self, list_of_nets, aggregator_fn, is_vector_env=False):
        super().__init__(list_of_nets, aggregator_fn, is_vector_env)

    def choose_action(self, state, greedy=False, prior=None):
        with torch.no_grad():
            
            if prior is None:
                prior = 1 / self.nA
            logprior = torch.log(torch.tensor(prior, device=self.device, dtype=torch.float32))
            # Get a sample from each net, then sample uniformly over them:
            logus = torch.stack([net.forward(state) * prior for net in self.nets], dim=1)
            logus = logus.squeeze(0)
            # Aggregate over the networks:
            logu, _ = self.aggregator_fn(logus, dim=0)

            if not self.is_vector_env:
                if greedy:
                    action_net_idx = torch.argmax(logu + logprior, dim=0)
                    # action = idxs[action_net_idx].cpu().numpy()
                    action = action_net_idx.cpu().numpy()
                else:
                    # pi* = pi0 * exp(logu)
                    logu = logu.clamp(-30,30)
                    in_exp = logu + logprior
                    in_exp -= (in_exp.max() + in_exp.min())/2
                    dist = torch.exp(in_exp)
                    dist /= torch.sum(dist)
                    c = Categorical(dist)
                    # action = idxs[c.sample().cpu().item()].cpu().numpy()
                    action = c.sample().cpu().numpy()
            else:
                raise NotImplementedError
                actions = np.array(actions)
                rnd_idx = np.expand_dims(np.random.randint(len(actions), size=actions.shape[1]), axis=0)
                action = np.take_along_axis(actions, rnd_idx, axis=0).squeeze(0)
            # perhaps re-weight this based on pessimism?
            return action
            # with torch.no_grad():
            #     logus = [net(state) for net in self.nets]
            #     logu = torch.stack(logus, dim=-1)
            #     logu = logu.squeeze(1)
            #     logu = torch.mean(logu, dim=-1)#[0]
            #     baseline = (torch.max(logu) + torch.min(logu))/2
            #     logu = logu - baseline
            #     logu = torch.clamp(logu, min=-20, max=20)
            #     dist = torch.exp(logu)
            #     dist = dist / torch.sum(dist)
            #     c = Categorical(dist)#, validate_args=True)
            #     return c.sample()#.item()

class OnlineUNets(OnlineNets):
    def __init__(self, list_of_nets, aggregator_fn, is_vector_env=False):
        super().__init__(list_of_nets, aggregator_fn, is_vector_env)

    def choose_action(self, state, greedy=False, prior=None):
        with torch.no_grad():
            
            if prior is None:
                prior = 1 / self.nA
            # Get a sample from each net, then sample uniformly over them:
            us = torch.stack([net.forward(state) * prior for net in self.nets], dim=1)
            us = us.squeeze(0)
            # Aggregate over the networks:
            u, _ = self.aggregator_fn(us, dim=0)

            if not self.is_vector_env:
                if greedy:
                    action_net_idx = torch.argmax(prior * u, dim=0)
                    action = action_net_idx.cpu().numpy()
                else:
                    dist = prior * u
                    dist /= torch.sum(dist)
                    c = Categorical(dist)
                    action = c.sample().cpu().numpy()
            else:
                raise NotImplementedError

            return action


class OnlineSoftQNets(OnlineNets):
    def __init__(self, list_of_nets, aggregator_fn, beta, is_vector_env=False):
        super().__init__(list_of_nets, aggregator_fn, is_vector_env)
        self.beta = beta
           
    def choose_action(self, state, greedy=False, prior=None):
        if prior is None:
            prior = 1 / self.nA
        with torch.no_grad():
            q_as = torch.stack([net.forward(state) for net in self], dim=1)
            q_as = q_as.squeeze(0)
            q_a, _ = self.aggregator_fn(q_as, dim=0)


            if greedy:
                action = torch.argmax(q_a).cpu().numpy()
            else:
                # pi propto e^beta Q:
                # first subtract a baseline from q_a:
                q_a = q_a - (torch.max(q_a) + torch.min(q_a))/2
                pi = prior * torch.exp(self.beta * q_a)
                pi = pi / torch.sum(pi)
                a = Categorical(pi).sample()
                action = a.cpu().numpy()
        return action

class LogUsa(nn.Module):
    def __init__(self, env, hidden_dim=256, device='cuda'):
        super(LogUsa, self).__init__()
        self.env = env
        self.device = device
        self.nS = get_flattened_obs_dim(self.env.observation_space)
        self.nA = get_action_dim(self.env.action_space)
        self.fc1 = nn.Linear(self.nS + self.nA, hidden_dim, device=self.device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, device=self.device)
        self.fc3 = nn.Linear(hidden_dim, 1, device=self.device)
        self.relu = nn.ReLU()

    def forward(self, obs, action):
        obs = torch.Tensor(obs).to(self.device)
        action = torch.Tensor(action).to(self.device)
        obs = preprocess_obs(obs, self.env.observation_space, normalize_images=NORMALIZE_IMG)
        x = torch.cat([obs, action], dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

LOG_SIG_MAX = 5
LOG_SIG_MIN = -30
epsilon = 1e-6

class Usa(nn.Module):
    def __init__(self, env, hidden_dim=256, device='cuda'):
        super(Usa, self).__init__()
        self.env = env
        self.device = device
        self.nS = get_flattened_obs_dim(self.env.observation_space)
        self.nA = get_action_dim(self.env.action_space)
        self.fc1 = nn.Linear(self.nS + self.nA, hidden_dim, device=self.device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, device=self.device)
        self.fc3 = nn.Linear(hidden_dim, 1, device=self.device)
        self.relu = nn.ReLU()

    def forward(self, obs, action):
        obs = torch.Tensor(obs).to(self.device)
        action = torch.Tensor(action).to(self.device)
        obs = preprocess_obs(obs, self.env.observation_space)
        x = torch.cat([obs, action], dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return nn.Softplus()(x)


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class GaussianPolicy(nn.Module):
    def __init__(self, hidden_dim, observation_space, action_space, use_action_bounds=False, device='cpu'):
        super(GaussianPolicy, self).__init__()
        self.device = device
        num_inputs = get_flattened_obs_dim(observation_space)
        num_actions = get_action_dim(action_space)
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)#, device=self.device)
            self.action_bias = torch.tensor(0.)#, device=self.device)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)#, device=self.device)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)#, device=self.device)
            
        self.observation_space = observation_space
        self.to(device)

    def forward(self, obs):
        obs = torch.Tensor(obs).to(self.device)
        obs = preprocess_obs(obs, self.observation_space, normalize_images=NORMALIZE_IMG)
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        # log_std = self.log_std_linear(x)
        # log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        log_std = torch.ones_like(mean) * -2
        return mean, log_std

    def sample(self, state):
        # with torch.no_grad():
        mean, log_std = self.forward(state)
        std = log_std.exp()
        # print(std)
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        noisy_action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return noisy_action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
    

class UNet(nn.Module):
    def __init__(self, env, device='cuda', hidden_dim=256, activation=nn.ReLU):
        super(UNet, self).__init__()
        self.using_vector_env = isinstance(env.action_space, gym.spaces.MultiDiscrete)
        self.env = env
        if self.using_vector_env:
            self.observation_space = self.env.single_observation_space
            self.action_space = self.env.single_action_space
        else:
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
        self.nA = self.action_space.n
        # do the check on an env before wrapping it
        self.is_image_space = is_image_space_simple(self.env.observation_space, self.using_vector_env)
        self.is_tabular = is_tabular(env)
        self.device = device
        model, nS = model_initializer(self.is_image_space,
                                  self.observation_space,
                                  self.nA,
                                  activation,
                                  hidden_dim,
                                  self.device)

        # Add a softplus layer:
        model = nn.Sequential(model, nn.Softplus())
        model.to(self.device)
        self.model = model
        self.nS = nS
        
        self.to(device)
     
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)  # Convert to PyTorch tensor
        
        # x = x.detach()
        x = preprocess_obs(x, self.env.observation_space)
        assert x.dtype == torch.float32, "Input must be a float tensor."

        # Reshape the image:
        if self.is_image_space:
            if len(x.shape) == 3:
                # Single image
                x = x.permute([2,0,1])
                x = x.unsqueeze(0)
            else:
                # Batch of images
                x = x.permute([0,3,1,2])
        elif self.is_tabular:
            # Single state
            if x.shape[0] == self.nS:
                x = x.unsqueeze(0)
            else: 
                x = x.squeeze(1)
                pass
        else:
            if len(x.shape) > len(self.nS):
                # in batch mode:
                pass
            else:
                # is a single state
                if isinstance(x.shape, torch.Size):
                    if x.shape == self.nS:
                        x = x.unsqueeze(0)
                else:
                    if (x.shape == self.nS).all():
                        x = x.unsqueeze(0)
                                
        x = self.model(x)
        # get machine epsilon:
        eps = torch.finfo(torch.float32).eps
        return x + eps
        # return x
        
    def choose_action(self, state, greedy=False, prior=None):
        if prior is None:
            prior = 1 / self.nA
        with torch.no_grad():
            # state = torch.tensor(state, device=self.device, dtype=torch.float32)  # Convert to PyTorch tensor
            u = self.forward(state)
            # prior = torch.tensor(prior.clone().detach(), device=self.device, dtype=torch.float32)
            # ensure prior is normalized:
            prior = prior / prior.sum()
            if greedy:
                # not worth exponentiating since it is monotonic
                a = (u * prior).argmax(dim=-1)
                return a.item()

            # First subtract a baseline:
            u = u / (torch.max(u) + torch.min(u))/2
            # clamp to avoid overflow:
            u = torch.clamp(u, min=1e-8, max=200)
            dist = u * prior
            dist = dist / torch.sum(dist)
            c = Categorical(dist)#, validate_args=True)
            # c = Categorical(logits=logu*prior)
            a = c.sample()

        return a.item()

    
class SoftQNet(torch.nn.Module):
    def __init__(self, env, device='cuda', hidden_dim=256, activation=nn.ReLU):
        super(SoftQNet, self).__init__()
        self.env = env
        self.nA = env.action_space.n
        self.is_image_space = is_image_space(env.observation_space)
        self.is_tabular = is_tabular(env)
        self.device = device
        # Start with an empty model:
        model = nn.Sequential()
        if self.is_image_space:
            raise NotImplementedError
        else:
            self.nS = env.observation_space.shape
            input_dim = self.nS[0]
            model.extend(nn.Sequential(
                nn.Linear(input_dim, hidden_dim, dtype=torch.float32),
                activation(),
                nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32),
                activation(),
                nn.Linear(hidden_dim, self.nA, dtype=torch.float32),    
            ))

        model.to(self.device)
        self.model = model
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)  # Convert to PyTorch tensor
        
        # x = x.detach()
        x = preprocess_obs(x, self.env.observation_space, normalize_images=NORMALIZE_IMG)
        assert x.dtype == torch.float32, "Input must be a float tensor."

        # Reshape the image:
        if self.is_image_space:
            if len(x.shape) == 3:
                # Single image
                x = x.permute([2,0,1])
                x = x.unsqueeze(0)
            else:
                # Batch of images
                x = x.permute([0,3,1,2])
        elif self.is_tabular:
            # Single state
            if x.shape[0] == self.nS:
                x = x.unsqueeze(0)
            else: 
                x = x.squeeze(1)
                pass
        else:
            if len(x.shape) > len(self.nS):
                # in batch mode:
                pass
            else:
                # is a single state
                if isinstance(x.shape, torch.Size):
                    if x.shape == self.nS:
                        x = x.unsqueeze(0)
                else:
                    if (x.shape == self.nS).all():
                        x = x.unsqueeze(0)

        x = self.model(x)
        return x

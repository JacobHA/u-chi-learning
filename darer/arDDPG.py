import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions.normal import Normal 
import itertools
import random
import os
from collections import deque

from BaseAgent import BaseAgent
from utils import logger_at_folder

def return_state_dim(spec):
    result = 0
    for i,k in enumerate(spec):
      result += spec[k].shape[0]
    return result
    
def process_state(obs):
    result= None
    for i,k in enumerate(obs):
      if(result is None):
          result = obs[k]
      else:
          result = np.concatenate([result,obs[k]])
          
    return result

class OUNoise(object):
    def __init__(self, action_spec, mu=0.0, theta=0.15, max_sigma=0.2, min_sigma=0.2, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_spec.shape[0]
        self.low          = action_spec.low
        self.high         = action_spec.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


class Buffer:
    def __init__(self, size = int(1e5)):
        self.state = deque(maxlen = size)
        self.reward = deque(maxlen = size)
        self.action = deque(maxlen = size)
        self.done = deque(maxlen = size)  
        self.next_state = deque(maxlen = size)
    
    def insert(self,state, reward, action, done, next_state):
        self.state.append(state)
        self.reward.append(reward)
        self.action.append(action)
        self.done.append(done)
        self.next_state.append(next_state)

    def sample(self,batch_size):
        index = random.sample(range(len(self.state)),batch_size)
        state_sample = [self.state[i] for i in index]
        
        reward_sample = [self.reward[i] for i in index]
        
        action_sample = [self.action[i] for i in index]
        
        done_sample = [self.done[i] for i in index]
        
        next_state_sample = [self.next_state[i] for i in index]

        
        return ( torch.tensor(np.array(state_sample)).float(), 
                torch.tensor(reward_sample).float(), 
                torch.tensor(np.array(action_sample)).float(), 
                torch.tensor(done_sample).float(), 
                torch.tensor(np.array(next_state_sample)).float())
     
    def __len__(self):
         return len(self.state)


class Actor(nn.Module):
    def __init__(self, env, act_func = nn.ReLU, lr = 3e-4, state_dim = None, 
                 action_dim = None, n_hidden_nodes = 256, device = 'auto'):
        super(Actor,self).__init__()

        self.env = env
        
        self.device = torch.device(device) if device != 'auto' else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if(state_dim is None):
           self.state_dim = env.observation_space.shape[0]
        else:
           self.state_dim = state_dim
        
        if(action_dim is None):
           self.action_dim = env.action_space.shape[0]
        else: 
           self.action_dim = action_dim 


        
        self.policy_net = nn.Sequential(nn.Linear(self.state_dim, n_hidden_nodes),
                                        act_func(),
                                        nn.Linear(n_hidden_nodes, n_hidden_nodes),
                                        act_func(),
                                        nn.Linear(n_hidden_nodes, self.action_dim))
                
        self.optimizer = Adam(self.parameters(),lr = lr)
        
    def forward(self,state):
        state = state.to(self.device)
        return self.policy_net(state).to('cpu')
            
    def get_action(self,state):
        
        action = self(state)        
        action = torch.tanh(action)
         
        return action
        
        
class Critic(nn.Module):
     def __init__(self, env, act_func = nn.ReLU, lr = 3e-4, state_dim = None, 
                  action_dim = None, n_hidden_nodes = 256, device = torch.device('cpu')):
        super(Critic, self).__init__()
         
        self.env = env
        
        self.device = device
        
        if(state_dim is None):
          self.state_dim = env.observation_space.shape[0]
        else:
            self.state_dim = state_dim
        
        if(action_dim is None):
           self.action_dim = env.action_space.shape[0]
        else:
            self.action_dim = action_dim



        self.q_net1 = nn.Sequential(nn.Linear(self.state_dim + self.action_dim, n_hidden_nodes),
                                    act_func(),
                                    nn.Linear(n_hidden_nodes, n_hidden_nodes),
                                    act_func(),
                                    nn.Linear(n_hidden_nodes, 1))
        
        self.q_net2 = nn.Sequential(nn.Linear(self.state_dim + self.action_dim, n_hidden_nodes),
                                    act_func(),
                                    nn.Linear(n_hidden_nodes, n_hidden_nodes),
                                    act_func(),
                                    nn.Linear(n_hidden_nodes, 1))
        
        self.optimizer = Adam(self.parameters(), lr = lr)
        
     def forward(self,state):
         state = state.to(self.device)
         return self.q_net1(state).to(torch.device('cpu')), self.q_net2(state).to(torch.device('cpu'))            
        
        
class Agent:
    def __init__(self,env,config, logger):
        
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        random.seed(config["seed"])
        
        self.env = env
        
        self.device = config['device']

        self.log = config['log']
        self.logger = logger

        self.state_dim = config['state_dim']
        
        self.action_dim = config['action_dim'] 
        
        self.actor = Actor(env, state_dim = self.state_dim, action_dim = self.action_dim,lr = config['lr_actor'], 
                           n_hidden_nodes = config['actor_hidden'], device = self.device)
        self.actor.to(self.device)
        self.actor_target = Actor(env, state_dim = self.state_dim, action_dim = self.action_dim,lr = config['lr_actor'],
                                  n_hidden_nodes = config['actor_hidden'], device = self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.to(self.device)

        self.critic = Critic(env, state_dim = self.state_dim, action_dim = self.action_dim,lr = config['lr_critic'], 
                             n_hidden_nodes = config['critic_hidden'],  device = self.device )
        self.critic.to(self.device)
        self.critic_target = Critic(env, state_dim = self.state_dim, action_dim = self.action_dim,lr = config['lr_critic'], 
                                    n_hidden_nodes = config['critic_hidden'], device = self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.to(self.device)
        
        self.rho = torch.tensor([0.0], requires_grad = True, dtype = torch.float)

        self.rho_optim = Adam([self.rho],lr = config['lr_rho'])

        self.tau = config['tau']

                   
      
    def update(self, batch, critic_freq = 1, actor_freq = 2):
        state, action, next_state, done, reward = batch

        
        if(self._n_updates % critic_freq == 0):

            with torch.no_grad():
                next_action = self.actor_target.get_action(next_state)                 
                q1, q2 = self.critic_target(torch.cat([next_state,next_action], axis = 1))
                temp = (1-done)*(torch.min(q1, q2).reshape(-1))
                q_target = reward + temp  
                        
            
            q_values = self.critic(torch.cat([state,action], axis = 1)) 
            critic_loss = 0.5 * sum([F.mse_loss(current_q.reshape(-1) + self.rho, q_target) for current_q in q_values])
            
            self.critic.optimizer.zero_grad()
            self.rho_optim.zero_grad()
            
            critic_loss.backward()
        
            self.critic.optimizer.step()
            self.rho_optim.step()


    
        if(self._n_updates % actor_freq == 0): 

            action_pi = self.actor.get_action(state)
                                    
            q1_pi, q2_pi = self.critic(torch.cat([state,action_pi], axis = 1))
            q_pi = torch.min(q1_pi, q2_pi).reshape(-1)
            actor_loss = -(q_pi).mean()
            

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
        
            with torch.no_grad():
                for params, params_target in zip(self.critic.parameters(), self.critic_target.parameters()):
                    params_target.data.mul_(self.tau)
                    params_target.data.add_((1 - self.tau)*params)

                for params, params_target in zip(self.actor.parameters(), self.actor_target.parameters()):
                    params_target.data.mul_(self.tau)
                    params_target.data.add_((1 - self.tau)*params)

            if(self.log):    
                self.logger.add_scalar("actor loss", actor_loss.item())
    
            
    def get_action(self, state):
        return self.actor.get_action(state)        

config = {
    'seed': 0,
    'lr_actor': 3e-4,
    'lr_critic': 3e-4,
    'lr_rho': 3e-4,
    'actor_hidden': 256,
    'critic_hidden': 256,
    'tau': 0.005,
    'state_dim': 8,
    'action_dim': 2,
    'buffer_size': 100_000,
    'batch_size': 256,
    'episodes': 1000,
    'warmup_samples': 1000,
    'critic_freq': 1,
    'actor_freq': 2,
    'epi_len': 1000,
    'epi_len_eval': 1000,
    'total_env_steps': 1_000_000,
    'eval_freq': 1000,
    'update_freq': 1,
    'device': 'cpu',
    'log': True,
}    

torch.manual_seed(config['seed'])
np.random.seed(config['seed'])

class arDDPG(BaseAgent):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.algo_name = 'arDDPG'
        
        self.logger = logger_at_folder(self.tensorboard_log,
                                       algo_name=f'{self.env_str}-{self.algo_name}{self.beta if self.beta == "auto" else ""}')
        
        
        self.noise = OUNoise(self.env.action_space)
        self.agent = Agent(self.env, config, logger = self.logger)


    def _initialize_networks(self):
        pass

    def exploration_policy(self, state):
        action = self.agent.get_action(torch.tensor(state).float())
        action = self.noise.get_action(action.detach().numpy())
        return action

    def evaluation_policy(self, state):
        self.actor.train(False)
        self.critic.train(False)                    
        action = self.actor.get_action(torch.tensor(state).float())

        return action
    
    def gradient_step(self, buffer, batch_size):
        self.agent.update(buffer, batch_size = batch_size, gradient_steps = 1, 
                          critic_freq = config['critic_freq'], 
                          actor_freq = config['actor_freq'])

# env = dmc2gym.make(config['domain'],config['task'], episode_length = config['epi_len'])
# env_eval = dmc2gym.make(config['domain'],config['task'], episode_length = config['epi_len_eval'])

# buffer = Buffer(config['buffer_size'])
# agent = Agent(env_eval, config)
# total_env_steps = config['total_env_steps']
# batch_size = config['batch_size']
# learning_start = config['warmup_samples']
# eval_freq = config['eval_freq']
# update_freq = config['update_freq']

# epi_len = config['epi_len']

def main():
    agent = arDDPG('Ant-v4', 1e-4, 1)
    agent.learn(total_timesteps=500_000)

if __name__ == '__main__':
    main()
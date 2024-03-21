from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape, get_flattened_obs_dim, preprocess_obs
import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional as F
import time
# import wandb
import sys
sys.path.append('darer')
from Models import OnlineUNets, Optimizers, TargetNets,  GaussianPolicy, Usa
from BaseAgent import BaseAgent
from utils import env_id_to_envs, log_class_vars, logger_at_folder
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.torch_layers import MlpExtractor, FlattenExtractor
from stable_baselines3.sac.policies import Actor

torch.backends.cudnn.benchmark = True
# raise warning level for debugger:
import warnings
warnings.filterwarnings("error")
class UActor(BaseAgent):
    def __init__(self,
                 *args,
                 actor_learning_rate: float = 1e-3,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.algo_name = 'UActor'

        self.n_samples = 1
        self.actor_learning_rate = actor_learning_rate
        self.nA = get_action_dim(self.env.action_space)        
        self.nS = get_flattened_obs_dim(self.env.observation_space)

        # Set up the logger:
        self.logger = logger_at_folder(self.tensorboard_log,
                                       algo_name=f'{self.env_str}-{self.algo_name}')
        self.log_hparams(self.logger)
        self._initialize_networks()

    def _initialize_networks(self):
        self.online_us = OnlineUNets([Usa(self.env,
                                               hidden_dim=self.hidden_dim,
                                               device=self.device)
                                        for _ in range(self.num_nets)],
                                        aggregator_fn=self.aggregator_fn)
        self.target_us = TargetNets([Usa(self.env,
                                               hidden_dim=self.hidden_dim,
                                               device=self.device)
                                        for _ in range(self.num_nets)])
        self.model = self.online_us
        self.target_us.load_state_dicts(
            [u.state_dict() for u in self.online_us])
        # self.actor = GaussianPolicy(self.hidden_dim, 
        #                             self.env.observation_space, self.env.action_space,
        #                             use_action_bounds=True,
        #                             device=self.device)
        self.actor = Actor(self.env.observation_space, self.env.action_space,
                    [self.hidden_dim, self.hidden_dim],
                    FlattenExtractor(self.env.observation_space),
                    self.nS,
                    )
        # send the actor to device:
        self.actor.to(self.device)
        # TODO: Try a fixed covariance network (no/ignored output)
        # Make (all) us and Actor learnable:
        opts = [torch.optim.Adam(u.parameters(), lr=self.learning_rate)
                for u in self.online_us]
        opts.append(torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_learning_rate))
        self.optimizers = Optimizers(opts)

    def gradient_descent(self, batch, grad_step):
        self.actor.set_training_mode(True)
    
        states, actions, next_states, dones, rewards = batch
        # actor_actions, curr_log_prob, means = self.actor.sample(states)
        actor_actions, curr_log_prob = self.actor.action_log_prob(states)

        online_curr_u = torch.stack([online_u(states, actions)
                                for online_u in self.online_us], dim=-1).squeeze(1)

        with torch.no_grad():
            pi_next_a, next_a_log_prob = self.actor.action_log_prob(next_states)

            # # sampled_action = torch.Tensor(np.array([self.env.action_space.sample() for 
            #                                 #   _ in range(self.batch_size)])).to(self.device)
            # # sampled_action, log_prob = self.actor.action_log_prob(observations)
            # # randomly shuffle the actions:
            # sampled_action = actions[torch.randperm(actions.shape[0])]
            # # sampled_action = actions#self.actor.action_log_prob(next_states)[0]#.squeeze(1)
            # # use same number of samples as the batch size for convenience:
            # # sampled_action = torch.Tensor(np.array([self.env.action_space.sample() for 
            # #                                _ in range(self.batch_size)])).to(self.device)
            # # use single sample for now:
            # # sampled_action = torch.Tensor(np.array([self.env.action_space.sample()])).to(self.device)
            # # repeat the sampled action for each state in batch:
            # # sampled_action = sampled_action.repeat(self.batch_size, 1)

            # # tile next states across the sampling amount (batch size):
            # next_states = next_states.unsqueeze(0)
            # next_states = next_states.repeat(self.batch_size,1,1)
            # # tile sampled actions across the amount of states (batch size), in 2nd slot:
            # sampled_action = sampled_action.unsqueeze(1)
            # sampled_action = sampled_action.repeat(1, self.batch_size, 1)
            # permute randomly:
            # sampled_action = sampled_action[torch.randperm(sel.shape[0])]

            ref_u_next = torch.stack([u(next_states, pi_next_a)
                                  for u in self.online_us], dim=0)
            # Aggregate over networks
            ref_u_next = self.aggregator_fn(ref_u_next, dim=0)

            # Calculate chi by summing over actions:
            # next_chi = ref_u_next.sum(dim=1) / self.batch_size

            curr_u = torch.stack([u(states, actions) for u in self.online_us], dim=0)
            curr_u = self.aggregator_fn(curr_u, dim=0)

            batch_rho = torch.mean(torch.exp(self.beta * rewards) * ref_u_next / curr_u)
            self.new_thetas[grad_step] = -torch.log(batch_rho) / self.beta

            target_next_u = torch.stack([target_u(next_states, pi_next_a)
                                            for target_u in self.target_us], dim=-1)

            next_u = self.aggregator_fn(target_next_u, dim=-1)
            # Need importance weighting:
            log_pi0 = -torch.log(torch.tensor(self.nA))
            next_u *= torch.exp(log_pi0 - next_a_log_prob).unsqueeze(1)
            # pi0(a'|s') / pi(a'|s')
            # next_u = next_u.sum(dim=1) / self.batch_size
            next_u = next_u * (1 - dones) #+ self.theta * dones

            expected_curr_u = torch.exp(self.beta * (rewards + self.theta)) * next_u
            expected_curr_u = expected_curr_u.squeeze(1)

        self.logger.record("train/theta", self.theta.item())
        self.logger.record("train/avg u", curr_u.mean().item())
        # Huber loss:
        loss = 0.5*sum(self.loss_fn(u, expected_curr_u) for u in online_curr_u.T)
        # MSE loss:
        actor_curr_u = torch.stack([online_u(states, actor_actions)
                                        for online_u in self.online_us], dim=-1)

        actor_log_curr_u = torch.log(self.aggregator_fn(actor_curr_u, dim=-1).squeeze())
        actor_loss = torch.mean(curr_log_prob - actor_log_curr_u)

        self.logger.record("train/log_prob", curr_log_prob.mean().item())
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/actor_loss", actor_loss.item())
        self.optimizers.zero_grad()
        # Increase update counter
        self._n_updates += self.gradient_steps

        # if self._n_updates % 100 == 0:
        # actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        
        # Clip gradient norm
        # loss.backward()
        # self.online_us.clip_grad_norm(self.max_grad_norm)

        return loss + actor_loss

    def exploration_policy(self, state: np.ndarray) -> (float, float):
        noisy_action, _ = self.actor.predict(state)
        kl = 0
        return noisy_action, kl

    def evaluation_policy(self, state: np.ndarray) -> float:
        self.actor.set_training_mode(False)

        # noisyaction, logprob, action = self.actor.sample(state)  # , deterministic=True)
        action, _ = self.actor.predict(state)#, deterministic=True)
        # action = action.cpu().numpy()
        return action
    
    def _update_target(self):
        # Do a Polyak update of parameters:
        self.target_us.polyak(self.online_us, self.tau)


def main():
    # env_id = 'LunarLanderContinuous-v2'
    # env_id = 'BipedalWalker-v3'
    # env_id = 'CartPole-v1'
    # env_id = 'Pendulum-v1'
    env_id = 'Hopper-v4'
    # env_id = 'HalfCheetah-v4'
    env_id = 'Ant-v4'
    # env_id = 'Simple-v0'
    from hparams import pendulum_logu as config
    agent = UActor(env_id, **config, device='cuda',
                      num_nets=2, tensorboard_log='pend', 
                      actor_learning_rate=5e-4, 
                      render=False, max_grad_norm=10, log_interval=1000,
                      )
                      
    agent.learn(total_timesteps=500_000)


if __name__ == '__main__':
    for _ in range(3):
        main()

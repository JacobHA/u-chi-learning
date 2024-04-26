from stable_baselines3.common.preprocessing import get_action_dim, get_flattened_obs_dim
import numpy as np
import torch
from torch.nn import functional as F

# import wandb
import sys
sys.path.append('darer')
from Models import LogUsa, OnlineUNets, Optimizers, TargetNets
from BaseAgent import BaseAgent
from utils import logger_at_folder
from stable_baselines3.common.torch_layers import MlpExtractor, FlattenExtractor
from stable_baselines3.sac.policies import Actor
import torch as th

class arSAC(BaseAgent):
    def __init__(self,
                 *args,
                 actor_learning_rate: float = 1e-3,
                 beta = 'auto',
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.algo_name = 'arSAC-nodone'

        self.actor_learning_rate = actor_learning_rate
        self.nA = get_action_dim(self.env.action_space)        
        self.nS = get_flattened_obs_dim(self.env.observation_space)

        # Set up the logger:
        self.logger = logger_at_folder(self.tensorboard_log,
                                       algo_name=f'{self.env_str}-{self.algo_name}{beta if beta == "auto" else ""}')
        self.log_hparams(self.logger)
        self.beta = beta
        self._initialize_networks()


    def _initialize_networks(self):
        self.online_critics = OnlineUNets([LogUsa(self.env,
                                               hidden_dim=self.hidden_dim,
                                               device=self.device)
                                        for _ in range(self.num_nets)],
                                        aggregator_fn=self.aggregator_fn)
        self.target_critics = TargetNets([LogUsa(self.env,
                                               hidden_dim=self.hidden_dim,
                                               device=self.device)
                                        for _ in range(self.num_nets)])
        self.model = self.online_critics
        self.target_critics.load_state_dicts(
            [q.state_dict() for q in self.online_critics])
        
        self.actor = Actor(self.env.observation_space, self.env.action_space,
                    [self.hidden_dim, self.hidden_dim],
                    FlattenExtractor(self.env.observation_space),
                    self.nS,
                    )
    
        self.prior = Actor(self.env.observation_space, self.env.action_space,
                           [self.hidden_dim, self.hidden_dim],
                           FlattenExtractor(self.env.observation_space),
                           self.nS,
                           )
            
        # send the actor to device:
        self.actor.to(self.device)
        self.prior.to(self.device)

        opts = [torch.optim.Adam(q.parameters(), lr=self.learning_rate)
                for q in self.online_critics]
        
        self.q_optimizers = Optimizers(opts)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                 lr=self.actor_learning_rate)
        self.prior_optimizer = torch.optim.Adam(self.prior.parameters(),
                                                    lr=self.actor_learning_rate / 1)

        
        
    def exploration_policy(self, state):
        self.actor.set_training_mode(False)
        # state = torch.tensor(state, dtype=torch.float32).to(self.device)
        # Get a stochastic action from the actor:
        action, _ = self.actor.predict(state)
        return action, 0
    
    def evaluation_policy(self, state):
        self.actor.set_training_mode(False)
        # Get a deterministic action from the actor:
        # state = torch.tensor(state, dtype=torch.float32)#.to(self.device)
        action, _ = self.actor.predict(state, deterministic=True)
        return action


    def gradient_descent(self, batch, grad_step):
        states, actions, next_states, dones, rewards = batch
        # ent_coef = self.beta ** (-1)

        optimizers = [self.actor_optimizer, self.q_optimizers, self.prior_optimizer]

        self.actor.set_training_mode(True)
        

        # We need to sample because `log_std` may have changed between two gradient steps
        # if self.use_sde:
        #     self.actor.reset_noise()

        # Action by the current actor for the sampled state
        actions_pi, log_prob = self.actor.action_log_prob(states)
        log_prob = log_prob.reshape(-1, 1)

        # TODO: Maybe can remove importance sampling now that the prior should approximate the optimal
        # _, prior_log_prob = self.prior.action_log_prob(states)
        # prior_log_prob = prior_log_prob.reshape(-1, 1)
            
        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        current_q_values = self.online_critics(states, actions)

        with th.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = self.actor.action_log_prob(next_states)
            # _, prior_next_log_prob = self.prior.action_log_prob(next_states)
            # Compute the next Q values: min over all critics targets
            next_q_values = th.cat(self.target_critics(next_states, next_actions), dim=1)
            next_q_values = self.aggregator_fn(next_q_values, dim=1)#, keepdim=True)
            # add entropy term
            # next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
            # td error + entropy term
            # target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            imp_sampling = 0#prior_next_log_prob.reshape(-1, 1) - next_log_prob.reshape(-1, 1)
            target_q_values = (self.beta) * (rewards - self.theta) + (next_q_values + imp_sampling) 

            min_q_values = th.cat(current_q_values, dim=1)
            min_q_values = self.aggregator_fn(min_q_values, dim=1)#, keepdim=True)

        new_theta = th.mean(rewards + (1/self.beta) * ((next_q_values + imp_sampling) - min_q_values))
        # self.theta += self.tau_theta * (new_theta - self.theta)
        self.new_thetas[grad_step] = new_theta

        # Compute critic loss
        critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
        assert isinstance(critic_loss, th.Tensor)  # for type checker
        # critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

        # Optimize the critic
        self.q_optimizers.zero_grad()
        critic_loss.backward()
        self.q_optimizers.step()

        # Min over all critic networks
        q_values_pi = th.cat(self.online_critics(states, actions_pi), dim=1)
        min_qf_pi = self.aggregator_fn(q_values_pi, dim=1)#, keepdim=True)
        actor_loss = (log_prob - min_qf_pi).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # with torch.no_grad():
        #     log_prob = self.actor.action_log_prob(states)[1]
        # # Optimize the prior (to match the actor):
        # prior_loss = (log_prob - prior_log_prob).mean()
        # self.prior_optimizer.zero_grad()
        # prior_loss.backward()
        # self.prior_optimizer.step()

    def _update_target(self):
        # TODO: Make sure we use gradient steps to track target updates:
        # if gradient_step % self.target_update_interval == 0:

        self.target_critics.polyak(self.online_critics, self.tau)
        # polyak_update(self.online_critics.parameters(), self.target_critics.parameters(), self.tau)
        # Copy running stats, see GH issue #996
        # polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)


def main():
    # env_id = 'LunarLanderContinuous-v2'
    # env_id = 'BipedalWalker-v3'
    # env_id = 'CartPole-v1'
    env_id = 'Pendulum-v1'
    # env_id = 'Hopper-v4'
    env_id = 'HalfCheetah-v4'
    # env_id = 'Ant-v4'
    # env_id = 'Simple-v0'
    from hparams import pendulum_logu as config
    # from simple_env import SimpleEnv
    agent = arSAC(env_id, **config, device='cuda',
                    num_nets=2, tensorboard_log='pend', 
                    actor_learning_rate=1e-4, 
                    render=False, max_grad_norm=10, log_interval=2000,
                      )
                      
    agent.learn(total_timesteps=500_000)


if __name__ == '__main__':
    for _ in range(10):
        main()

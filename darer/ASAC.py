from stable_baselines3.common.preprocessing import get_action_dim, get_flattened_obs_dim
import numpy as np
import torch
from torch.nn import functional as F

# import wandb
import sys
sys.path.append('darer')
from Models import Qsa, OnlineUNets, Optimizers, TargetNets
from BaseAgent import BaseAgent
from utils import logger_at_folder
from stable_baselines3.common.torch_layers import MlpExtractor, FlattenExtractor
from stable_baselines3.sac.policies import Actor
import torch as th


# torch.backends.cudnn.benchmark = True
# raise warning level for debugger:
# import warnings
# warnings.filterwarnings("error")
class ASAC(BaseAgent):
    def __init__(self,
                 *args,
                 actor_learning_rate: float = 1e-3,
                #  beta = 'auto',
                 use_rawlik: bool = False,
                 use_dones: bool = True,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.algo_name = f'ASAC-' + 'no'*(not use_dones) + 'auto'*(self.beta == 'auto') + 'max'
        self.use_dones = use_dones
        self.use_rawlik = use_rawlik
        self.actor_learning_rate = actor_learning_rate
        self.nA = get_action_dim(self.env.action_space)        
        self.nS = get_flattened_obs_dim(self.env.observation_space)

        # Set up the logger:
        self.logger = logger_at_folder(self.tensorboard_log,
                                       algo_name=f'{self.env_str}-{self.algo_name}{self.beta if self.beta == "auto" else ""}')
        self.log_hparams(self.logger)
        self.logpi0 = th.log(th.tensor(1/self.nA, device=self.device))
        # self.ent_coef_optimizer: Optional[th.optim.Adam] = None
        if self.beta != 'auto':
            self.ent_coef = self.beta**(-1)
        else:
            self.ent_coef = 'auto'
        self._initialize_networks()
        self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))


    def _initialize_networks(self):
        self.online_critics = OnlineUNets([Qsa(self.env,
                                               hidden_dim=self.hidden_dim,
                                               device=self.device)
                                        for _ in range(self.num_nets)],
                                        aggregator_fn=self.aggregator_fn)
        self.target_critics = TargetNets([Qsa(self.env,
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
            
        # send the actor to device:
        self.actor.to(self.device)
        # TODO: Try a fixed covariance network (no/ignored output)
        opts = [torch.optim.Adam(q.parameters(), lr=self.learning_rate)
                for q in self.online_critics]
        
        self.q_optimizers = Optimizers(opts)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                 lr=self.actor_learning_rate)

        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef])#, lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)
            self.ent_coef_optimizer = None
        
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

        optimizers = [self.actor_optimizer, self.q_optimizers]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        # self._update_learning_rate(optimizers)

        self.actor.set_training_mode(True)

        # We need to sample because `log_std` may have changed between two gradient steps
        # if self.use_sde:
        #     self.actor.reset_noise()

        # Action by the current actor for the sampled state
        actions_pi, log_prob = self.actor.action_log_prob(states)
        log_prob = log_prob.reshape(-1, 1)

        ent_coef_loss = None
        if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            ent_coef = th.exp(self.log_ent_coef.detach())
            # ent_coef_loss = -((log_prob + self.target_entropy).detach()).mean()
            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()

            # ent_coef_losses.append(ent_coef_loss.item())
        else:
            ent_coef = self.ent_coef_tensor

        # ent_coefs.append(ent_coef.item())

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
            
        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        current_q_values = self.online_critics(states, actions)

        with th.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = self.actor.action_log_prob(next_states)
            # Compute the next Q values: min over all critics targets
            next_q_values = th.cat(self.target_critics(next_states, next_actions), dim=1)
            next_q_values = self.aggregator_fn(next_q_values, dim=1) 
            # add entropy term
            next_v_values = next_q_values - ent_coef * (next_log_prob.reshape(-1, 1) - self.logpi0)
            # td error + entropy term
            # target_q_values = rewards +  * self.gamma * next_q_values
            if self.use_dones:
                next_v_values = next_v_values * (1 - dones)

            target_q_values = rewards - self.theta + next_v_values 

            min_q_values = th.cat(current_q_values, dim=1)
            min_q_values = self.aggregator_fn(min_q_values, dim=1)

            # new_theta = th.mean(rewards + next_v_values - min_q_values)
            new_theta = th.mean(rewards - ent_coef * (log_prob.reshape(-1, 1) - self.logpi0))
            self.logger.record(f"train/new_theta", new_theta.item())

        self.theta += self.tau_theta * (new_theta - self.theta)

        # Compute critic loss
        critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
        assert isinstance(critic_loss, th.Tensor)  # for type checker


        self.lr = self.q_optimizers.get_lr()
        # Optimize the critic
        self.q_optimizers.zero_grad()
        critic_loss.backward()
        self.q_optimizers.step()

        # Compute actor loss
        # Min over all critic networks
        q_values_pi = th.cat(self.online_critics(states, actions_pi), dim=1)
        min_qf_pi = self.aggregator_fn(q_values_pi, dim=1)
        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # log newest temperature:
        self.logger.record("train/temp", ent_coef.item())
  

    def _update_target(self):
        # TODO: Make sure we use gradient steps to track target updates:
        # if gradient_step % self.target_update_interval == 0:

        self.target_critics.polyak(self.online_critics, self.tau)
        # polyak_update(self.online_critics.parameters(), self.target_critics.parameters(), self.tau)
        # Copy running stats, see GH issue #996
        # polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)


    def _update_prior(self):
        if self.use_rawlik:
            # Polyak average the prior:
            self.target_prior.polyak(self.online_prior, self.tau)


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
    agent = ASAC(env_id, **config, device='cuda',
                    num_nets=2, tensorboard_log='pend', 
                    actor_learning_rate=1e-4, 
                    render=False, max_grad_norm=10, log_interval=2000,
                      )
                      
    agent.learn(total_timesteps=500_000)


if __name__ == '__main__':
    for _ in range(10):
        main()

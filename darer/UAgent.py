
import time
from typing import Optional
import numpy as np
import torch
from torch.nn import functional as F
from BaseAgent import BaseAgent
from Models import OnlineUNets, Optimizers, TargetNets, UNet, PiNet
from utils import logger_at_folder


class UAgent(BaseAgent):
    def __init__(self,
                 *args,
                 use_rawlik=False,
                 prior_update_interval: int = 1_000,
                 prior_tau: float = 0.9,
                 name: Optional[str] = None,
                 **kwargs,
                 ):
        self.algo_name = 'U' if name is None else name
        if use_rawlik:
            self.algo_name += '-rawlik'
        self.use_rawlik = use_rawlik
        super().__init__(*args, **kwargs)

        # Set up the logger:
        self.logger = logger_at_folder(self.tensorboard_log,
                                       algo_name=f'{self.env_str}-{self.algo_name}')
        self.log_hparams(self.logger)
        self._initialize_networks()

    def _initialize_networks(self):
        self.online_us = OnlineUNets([UNet(self.env, 
                                          hidden_dim=self.hidden_dim, 
                                          device=self.device,
                                          activation=torch.nn.ReLU,
                                          use_rawlik=self.use_rawlik
                                        )
                                     for _ in range(self.num_nets)],
                                    aggregator_fn=self.aggregator_fn)
        # alias for compatibility as self.model:
        self.model = self.online_us
        self.target_us = TargetNets([UNet(self.env, 
                                            hidden_dim=self.hidden_dim,
                                            device=self.device,
                                            use_rawlik=self.use_rawlik)
                                     for _ in range(self.num_nets)])
        self.target_us.load_state_dicts(
            [u.state_dict() for u in self.online_us])
        # Make (all) Us learnable:
        opts = [torch.optim.Adam(u.parameters(), lr=self.learning_rate)
                for u in self.online_us]
        
        self.optimizers = Optimizers(opts, self.scheduler_str)

    def exploration_policy(self, state: np.ndarray) -> (int, float):
        with torch.no_grad():

            # return self.env.action_space.sample()
            # u, pi0 = self.online_us.forward(state)
            
            chosen_action = self.online_us.choose_action(state, greedy=False)
            # pi_by_pi0 = self.aggregator_fn(torch.stack(
            #     [u(state) for u in self.online_us], dim=0), dim=0)
            # # kl = pi_by_pi0[0][chosen_action] * pi0[chosen_action] * torch.log(pi_by_pi0[0][chosen_action])
            # kl = (pi_by_pi0[0] * pi0* torch.log(pi_by_pi0[0])).sum()

            # kl = float(kl.item())
            kl = 0
            # chosen_action = self.env.action_space.sample()
            return chosen_action, kl

    def evaluation_policy(self, state: np.ndarray) -> int:
        with torch.no_grad():
            return self.online_us.choose_action(state, greedy=True)

    def _update_target(self):
        # Do a Polyak update of parameters:
        self.target_us.polyak(self.online_us, self.tau)

    def _on_step(self):
        super()._on_step()
        #TODO: put this in _update_target
        # if self.use_rawlik:
        #     if self.env_steps % self.prior_update_interval == 0:
        #         self.target_prior.polyak(self.online_prior, self.prior_tau)

    def gradient_descent(self, batch, grad_step: int):
        states, actions, next_states, dones, rewards = batch
        # rewards[dones.bool()] -= 1
        # Calculate the current u values (feedforward):
        # curr_ua = torch.stack([online_u(states).squeeze()
        #                     for online_u in self.online_us], dim=1)
        curr_u, curr_pi0 = self.online_us.forward(states)
        curr_ua = curr_u.squeeze(-1).gather(1, actions.long())
        curr_priora = curr_pi0.squeeze(-1)
        

        with torch.no_grad():
            online_u_next, online_pi0_next = self.online_us.forward(next_states)
            online_curr_u, online_curr_pi0 = self.online_us.forward(states)
            online_curr_ua = online_curr_u.squeeze(-1).gather(1, actions.long())
            #TODO: collapse these into one call
            # online_curr_ua = torch.stack([u(states)[0]
            #                               for u in self.online_us], dim=0)
            
            target_ua, target_priora = torch.stack([target_upi.forward(states)
                                                    for target_upi in self.target_us], dim=-1)
            # print(target_ua.shape)
            target_ua = self.aggregator_fn(target_ua.squeeze(-1), dim=-1)
            # target_priora = target_priora.squeeze(-1)
            target_priora = torch.mean(target_priora, dim=-1)
            
            # print(online_pi0_next.shape)
            # exit()
            online_chi = (online_u_next * online_pi0_next).sum(dim=-1)
            online_curr_ua = online_curr_ua.squeeze(-1)
            in_log = online_chi / online_curr_ua
            # print(in_log.shape)
            # in_log = torch.clamp(in_log, min=1e-6, max=1e6)
            # clamp to a tolerable range:
            # batch_theta = -torch.mean(rewards.squeeze(-1) + torch.log(in_log) / self.beta, dim=1)
            batch_rho = torch.mean(torch.exp(self.beta * rewards.squeeze(-1)) * in_log)

            # batch_theta = torch.clamp(batch_theta, min=-30, max=30)
            self.new_thetas[grad_step, :] = -torch.log(batch_rho) / self.beta
            
            target_next_us, target_next_priors = torch.stack([targets.forward(next_states)
                                                                for targets in self.target_us])

            # logsumexp over actions:
            target_next_u = self.aggregator_fn(target_next_us, dim=0)
            target_next_prior = torch.mean(target_next_priors, dim=0)
            next_chis = (target_next_u * target_next_prior).sum(dim=-1)

            next_chis = next_chis.reshape(-1,1)
            assert next_chis.shape == dones.shape
            next_chis = next_chis * (1-dones)  # + 1 * dones

            # "Backup" eigenvector equation:
            in_exp = rewards + self.theta
            in_exp = torch.clamp(in_exp, max=30)
            expected_curr_u = torch.exp(self.beta * (in_exp)) * next_chis
            expected_curr_u = expected_curr_u.squeeze()

        # self.logger.record("train/u-avg", torch.mean(curr_u, dim=0))

        # Calculate the u ("critic") loss:
        # print(curr_u.shape)
        # print(expected_curr_u.shape)
        loss = 0.5*sum(self.loss_fn(u, expected_curr_u) for u in curr_u.permute(1, 0))
        self.logger.record("train/loss", loss.item())
        if self.use_rawlik:
            # prior_loss = self.loss
            # _fn(curr_prior.squeeze(), self.aggregator_fn(online_curr_u,dim=0)[0])
            # print(online_curr_u.shape)
            # exit()
            pistar = online_curr_u * online_curr_pi0
            pistar /= pistar.sum(dim=-1)

            prior_loss = F.kl_div(curr_priora.log(),
                                    pistar,
                                    reduction='batchmean',
                                    log_target=False)

            self.logger.record("train/prior_loss", prior_loss.item())
            # prior_loss.backward()
            loss += prior_loss

        # weight = 1e-1
        # Mean across actions:
        # curr_umean = torch.mean(curr_ua, dim=-1, keepdim=True).repeat(1, 1, self.nA)
        # regularize_loss = weight * torch.nn.functional.l1_loss(curr_ua, curr_umean)
        # self.logger.record("train/regularize_loss", regularize_loss.item())
        # print which params have grad:
        return loss #+ regularize_loss


def main():
    from disc_envs import get_environment
    env_id = get_environment('Pendulum21', nbins=3,
                             max_episode_steps=200, reward_offset=0)

    env_id = 'CartPole-v1'
    # env_id = 'Taxi-v3'
    # env_id = 'CliffWalking-v0'
    env_id = 'Acrobot-v1'
    # env_id = 'LunarLander-v2'
    # env_id = 'PongNoFrameskip-v4'
    # env_id = 'BreakoutNoFrameskip-v4'
    # env_id = 'FrozenLake-v1'
    # env_id = 'MountainCar-v0'
    # env_id = 'Drug-v0'

    from hparams import acrobot_u as config

    agent = UAgent(env_id, **config, device='cuda', log_interval=500,
                   tensorboard_log='pong', num_nets=2, render=True, #aggregator='min',
                   beta_schedule='none', use_rawlik=True,
                   beta_end=5)
    #    scheduler_str='none')  # , beta_schedule='none', beta_end=2.4,
    # use_rawlik=True)
    # Measure the time it takes to learn:
    t0 = time.thread_time_ns()
    agent.learn(total_timesteps=10_000)
    t1 = time.thread_time_ns()
    print(f"Time to learn: {(t1-t0)/1e9} seconds")


if __name__ == '__main__':

    for _ in range(1):
        main()

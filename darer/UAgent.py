
import time
import numpy as np
import torch
from torch.nn import functional as F
from BaseAgent import BaseAgent
from Models import OnlineUNets, Optimizers, TargetNets, UNet
from utils import logger_at_folder


class UAgent(BaseAgent):
    def __init__(self,
                 *args,
                 use_rawlik=False,
                 prior_update_interval: int = 20_000,
                 prior_tau: float = 1.0,
                 **kwargs,
                 ):
        self.algo_name = 'U'
        self.prior_update_interval = prior_update_interval
        self.prior_tau = prior_tau
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
                                          device=self.device)
                                     for _ in range(self.num_nets)],
                                    aggregator_fn=self.aggregator_fn)
        # alias for compatibility as self.model:
        self.model = self.online_us
        self.target_us = TargetNets([UNet(self.env, hidden_dim=self.hidden_dim, device=self.device)
                                     for _ in range(self.num_nets)])
        self.target_us.load_state_dicts(
            [u.state_dict() for u in self.online_us])
        # Make (all) Us learnable:
        opts = [torch.optim.Adam(u.parameters(), lr=self.learning_rate)
                for u in self.online_us]
        if self.use_rawlik:
            self.online_prior = OnlineUNets(
                [UNet(self.env, hidden_dim=self.hidden_dim, device=self.device)],
                aggregator_fn=self.aggregator_fn)
            self.target_prior = TargetNets(
                [UNet(self.env, hidden_dim=self.hidden_dim, device=self.device)])
            self.target_prior.load_state_dicts(
                [u.state_dict() for u in self.online_prior])
            opts.append(torch.optim.Adam(
                self.online_prior.nets[0].parameters(), lr=self.learning_rate))

        self.optimizers = Optimizers(opts, self.scheduler_str)

    def exploration_policy(self, state: np.ndarray) -> (int, float):
        with torch.no_grad():

            # return self.env.action_space.sample()
            if self.use_rawlik:
                pi0 = self.target_prior.nets[0](state).squeeze()
                pi0 /= pi0.sum()
            else:
                pi0 = 1 / self.nA
            chosen_action = self.online_us.choose_action(state, prior=pi0, greedy=False)
            pi_by_pi0 = self.aggregator_fn(
                torch.stack([u(state) for u in self.online_us], dim=0), dim=0)[0]
            kl = torch.log(pi_by_pi0[0][chosen_action])
            kl = float(kl.item())
            return chosen_action, kl

    def evaluation_policy(self, state: np.ndarray) -> int:
        with torch.no_grad():
            if self.use_rawlik:
                pi0 = self.target_prior.nets[0](state).squeeze()
                pi0 /= pi0.sum()
            else:
                pi0 = None

            return self.online_us.choose_action(state, prior=pi0, greedy=False)

    def _update_target(self):
        # Do a Polyak update of parameters:
        self.target_us.polyak(self.online_us, self.tau)

    def _on_step(self):
        super()._on_step()
        if self.use_rawlik:
            if self.env_steps % self.prior_update_interval == 0:
                self.target_prior.polyak(self.online_prior, self.prior_tau)

    def gradient_descent(self, batch, grad_step: int):
        # Sample a batch from the replay buffer:
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, next_states, dones, rewards = batch
        # rewards[dones.bool()] -= 10
        # Calculate the current u values (feedforward):
        curr_u = torch.cat([online_u(states).squeeze().gather(1, actions.long())
                            for online_u in self.online_us], dim=1)
        if self.use_rawlik:
            curr_priora = self.online_prior.nets[0](
                states).squeeze()  # .gather(1, actions.long())
            # normalize the prior:
            curr_priora /= curr_priora.sum(dim=-1, keepdim=True)


        with torch.no_grad():
            online_u_next = torch.stack([u(next_states)
                                         for u in self.online_us], dim=0)
            online_curr_u = torch.stack([u(states).gather(1, actions)
                                         for u in self.online_us], dim=0)
            online_curr_ua = torch.stack([u(states)
                                          for u in self.online_us], dim=0)
            if self.use_rawlik:
                target_priora = self.target_prior.nets[0](states).squeeze()
                target_priora /= target_priora.sum(dim=-1, keepdim=True)

                target_prior_next = self.target_prior.nets[0](
                    next_states).squeeze()
                target_prior_next /= target_prior_next.sum(
                    dim=-1, keepdim=True)

            else:
                target_priora = torch.ones_like(
                    actions, device=self.device) * (1/self.nA)
                target_prior_next = torch.ones_like(
                    actions, device=self.device) * (1/self.nA)

            online_chi = (
                online_u_next * target_prior_next.repeat(self.num_nets, 1, 1)).sum(dim=-1)
            online_curr_u = online_curr_u.squeeze(-1)
            in_log = online_chi / online_curr_u
            # clamp to a tolerable range:
            self.new_thetas[grad_step, :] = - \
                (torch.mean(rewards.squeeze(-1) + torch.log(in_log) / self.beta, dim=1))

            target_next_us = [target_u(next_states)
                              for target_u in self.target_us]

            # logsumexp over actions:
            target_next_us = torch.stack(target_next_us, dim=1)
            next_us = (target_next_us * target_prior_next.unsqueeze(1).repeat(1,
                       self.num_nets, 1)).sum(dim=-1)
            next_u, _ = self.aggregator_fn(next_us, dim=1)

            next_u = next_u.reshape(-1, 1)
            assert next_u.shape == dones.shape
            next_u = next_u * (1-dones)  # + 1 * dones

            # "Backup" eigenvector equation:
            in_exp = rewards + self.theta
            expected_curr_u = torch.exp(self.beta * (in_exp)) * next_u
            expected_curr_u = expected_curr_u.squeeze(1)

        # Calculate the u ("critic") loss:
        loss = 0.5*sum(self.loss_fn(u, expected_curr_u) for u in curr_u.T)
        if self.use_rawlik:
            # prior_loss = self.loss_fn(curr_prior.squeeze(), self.aggregator_fn(online_curr_u,dim=0)[0])
            pistar = self.aggregator_fn(online_curr_ua, dim=0)[
                0] * target_priora
            pistar /= pistar.sum(dim=-1, keepdim=True)

            prior_loss = F.kl_div(curr_priora.squeeze().log(
            ), pistar, reduction='batchmean', log_target=False)

            self.logger.record("train/prior_loss", prior_loss.item())
            prior_loss.backward()

        return loss


def main():
    from disc_envs import get_environment
    env_id = get_environment('Pendulum21', nbins=3,
                             max_episode_steps=200, reward_offset=0)

    env_id = 'CartPole-v1'
    # env_id = 'Taxi-v3'
    # env_id = 'CliffWalking-v0'
    # env_id = 'Acrobot-v1'
    # env_id = 'LunarLander-v2'
    env_id = 'PongNoFrameskip-v4'
    # env_id = 'FrozenLake-v1'
    # env_id = 'MountainCar-v0'
    # env_id = 'Drug-v0'

    from hparams import nature_pong as config

    agent = UAgent(env_id, **config, device='cuda', log_interval=2500,
                   tensorboard_log='pong', num_nets=2, render=False, aggregator='min',
                   beta_schedule='linear', beta_end=5)  # , use_rawlik=True)
    #    scheduler_str='none')  # , beta_schedule='none', beta_end=2.4,
    # use_rawlik=True)
    # Measure the time it takes to learn:
    t0 = time.thread_time_ns()
    agent.learn(total_timesteps=1_500_000)
    t1 = time.thread_time_ns()
    print(f"Time to learn: {(t1-t0)/1e9} seconds")


if __name__ == '__main__':

    for _ in range(1):
        main()

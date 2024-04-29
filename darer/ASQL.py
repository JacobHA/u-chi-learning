import numpy as np
import torch
from torch.nn import functional as F
from BaseAgent import BaseAgent
from Models import QNet, OnlineQNets, OnlineUNets, Optimizers, PiNet, TargetNets
from utils import logger_at_folder

class ASQL(BaseAgent):
    def __init__(self,
                 *args,
                 use_ppi=False,
                 prior_update_interval: int = 1_000,
                 prior_tau: float = 0.9,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.algo_name = 'ASQL' + ('-PPI' if use_ppi else '')
        self.use_ppi = use_ppi
        self.prior_update_interval = prior_update_interval
        self.prior_tau = prior_tau
        # Set up the logger:
        self.logger = logger_at_folder(self.tensorboard_log,
                                       algo_name=f'{self.env_str}-{self.algo_name}')
        self.log_hparams(self.logger)
        self._initialize_networks()

    def _initialize_networks(self):
        self.online_qs = OnlineQNets([QNet(self.env,
                                           hidden_dim=self.hidden_dim, 
                                           device=self.device)
                                        for _ in range(self.num_nets)],
                                        aggregator_fn=self.aggregator_fn)
        # alias for compatibility as self.model:
        self.model = self.online_qs

        self.target_qs = TargetNets([QNet(self.env, 
                                                hidden_dim=self.hidden_dim, 
                                                device=self.device)
                                        for _ in range(self.num_nets)])
        self.target_qs.load_state_dicts(
            [q.state_dict() for q in self.online_qs])
        # Make (all) q+s learnable:
        opts = [torch.optim.Adam(q.parameters(), lr=self.learning_rate)
                for q in self.online_qs]

        if self.use_ppi:
            self.online_prior = OnlineUNets(
                [PiNet(self.env, hidden_dim=self.hidden_dim, device=self.device)],
                aggregator_fn=self.aggregator_fn)
            self.target_prior = TargetNets(
                [PiNet(self.env, hidden_dim=self.hidden_dim, device=self.device)])
            self.target_prior.load_state_dicts(
                [u.state_dict() for u in self.online_prior])
            opts.append(torch.optim.Adam(
                self.online_prior.nets[0].parameters(), lr=self.learning_rate))

        self.optimizers = Optimizers(opts, self.scheduler_str)

    def exploration_policy(self, state: np.ndarray) -> (int, float):
        # return self.env.action_space.sample(), 0
        kl = 0
        pi0 = None

        if self.use_ppi:
            pi0 = self.target_prior.nets[0](state).squeeze()

        return self.online_qs.choose_action(state, self.beta, greedy=False, prior=pi0), kl

    def evaluation_policy(self, state: np.ndarray) -> int:
        pi0 = None
        if self.use_ppi:
            pi0 = self.target_prior.nets[0](state).squeeze()
                
        return self.online_qs.choose_action(state, self.beta, greedy=True, prior=pi0)

    def gradient_descent(self, batch, grad_step: int):
        states, actions, next_states, dones, rewards = batch
        if self.use_ppi:
            curr_priora = self.online_prior.nets[0](states).squeeze()

        # Calculate the current q values (feedforward):
        curr_q = torch.cat([online_q(states).squeeze().gather(1, actions.long())
                               for online_q in self.online_qs], dim=1)

        with torch.no_grad():
            online_q_next = torch.stack([q(next_states)
                                            for q in self.online_qs], dim=0)
            online_curr_q = torch.stack([q(states).gather(1, actions)
                                            for q in self.online_qs], dim=0)

            if self.use_ppi:
                target_priora = self.target_prior.nets[0](states).squeeze()
                # target_priora /= target_priora.sum(dim=-1, keepdim=True)

                target_prior_next = self.target_prior.nets[0](next_states).squeeze()
                target_prior_next /= target_prior_next.sum(dim=-1, keepdim=True)

            else:
                target_priora = torch.ones(self.batch_size, self.nA, device=self.device) * (1/self.nA)
                target_prior_next = torch.ones(self.batch_size, self.nA, device=self.device) * (1/self.nA)

            log_prior_next = torch.log(target_prior_next + 1e-8)

            # Aggregate the qs:
            online_q_next = self.aggregator_fn(online_q_next, dim=0).squeeze(0)
            online_curr_q = self.aggregator_fn(online_curr_q, dim=0).squeeze(0)
            online_next_v = self.beta**(-1) * torch.logsumexp(self.beta * online_q_next + log_prior_next.repeat(1, 1), 
                                                         dim=-1, 
                                                         keepdim=True)

            # new_theta = -torch.mean( rewards + (online_log_chi - online_curr_q) / self.beta, dim=0)
            # new_theta = -torch.mean(rewards + self.beta**(-1) * online_next_v - online_curr_q , dim=0)
            new_theta = torch.mean(rewards - online_curr_q + online_next_v, dim=0)


            self.theta = (1 - self.tau_theta) * self.theta + self.tau_theta * new_theta

            target_next_qs = [target_q(next_states)
                                 for target_q in self.target_qs]

            # logsumexp over actions:
            target_next_qs = torch.stack(target_next_qs, dim=1)
            target_next_q = self.aggregator_fn(target_next_qs, dim=1).squeeze(1)
            next_v = self.beta**(-1) * torch.logsumexp(self.beta * target_next_q + log_prior_next, dim=-1)

            next_v = next_v.reshape(-1, 1)
            assert next_v.shape == dones.shape
            next_v = next_v * (1 - dones)  # + self.theta * dones

            # "Backup" eigenvector equation:
            expected_curr_q = rewards - self.theta + next_v
            expected_curr_q = expected_curr_q.squeeze(1)

        # Calculate the q ("critic") loss:
        loss = 0.5*sum(self.loss_fn(q, expected_curr_q) for q in curr_q.T)
        
        if self.use_ppi:
            # prior_loss = self.loss_fn(curr_prior.squeeze(), self.aggregator_fn(online_curr_u,dim=0)[0])
            pistar = torch.exp(self.beta * self.aggregator_fn(online_curr_q, dim=0)) * target_priora
            pistar /= pistar.sum(dim=-1, keepdim=True)

            prior_loss = F.kl_div(curr_priora.squeeze().log(), pistar, reduction='batchmean', log_target=False)

            self.logger.record("train/prior_loss", prior_loss.item())
            # prior_loss.backward()
            loss += prior_loss

        self.optimizers.zero_grad()
        # Clip gradient norm
        loss.backward()
        self.model.clip_grad_norm(self.max_grad_norm)
        self.optimizers.step()
        return None 

    def _update_target(self):
        # Do a Polyak update of parameters:
        self.target_qs.polyak(self.online_qs, self.tau)


    def _update_prior(self):
        # Update the prior:
        if self.use_ppi:
            self.target_prior.polyak(self.online_prior, self.prior_tau)
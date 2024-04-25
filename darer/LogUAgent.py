import numpy as np
import torch
from torch.nn import functional as F
from BaseAgent import BaseAgent
from Models import LogUNet, OnlineLogUNets, OnlineUNets, Optimizers, PiNet, TargetNets
from utils import logger_at_folder

class LogUAgent(BaseAgent):
    def __init__(self,
                 *args,
                 use_rawlik=False,
                 prior_update_interval: int = 1_000,
                 prior_tau: float = 0.9,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.algo_name = 'LogU' + ('-PPI' if use_rawlik else '')
        self.use_rawlik = use_rawlik
        self.prior_update_interval = prior_update_interval
        self.prior_tau = prior_tau
        # Set up the logger:
        self.logger = logger_at_folder(self.tensorboard_log,
                                       algo_name=f'{self.env_str}-{self.algo_name}')
        self.log_hparams(self.logger)
        self._initialize_networks()

    def _initialize_networks(self):
        self.online_logus = OnlineLogUNets([LogUNet(self.env,
                                                hidden_dim=self.hidden_dim, 
                                                device=self.device)
                                        for _ in range(self.num_nets)],
                                       aggregator_fn=self.aggregator_fn)
        # alias for compatibility as self.model:
        self.model = self.online_logus

        self.target_logus = TargetNets([LogUNet(self.env, 
                                                hidden_dim=self.hidden_dim, 
                                                device=self.device)
                                        for _ in range(self.num_nets)])
        self.target_logus.load_state_dicts(
            [logu.state_dict() for logu in self.online_logus])
        # Make (all) LogUs learnable:
        opts = [torch.optim.Adam(logu.parameters(), lr=self.learning_rate)
                for logu in self.online_logus]

        if self.use_rawlik:
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

        if self.use_rawlik:
            pi0 = self.target_prior.nets[0](state).squeeze()

        return self.online_logus.choose_action(state, greedy=False, prior=pi0), kl

    def evaluation_policy(self, state: np.ndarray) -> int:
        pi0 = None
        if self.use_rawlik:
            pi0 = self.target_prior.nets[0](state).squeeze()
                
        return self.online_logus.choose_action(state, greedy=True, prior=pi0)

    def gradient_descent(self, batch, grad_step: int):
        states, actions, next_states, dones, rewards = batch
        if self.use_rawlik:
            curr_priora = self.online_prior.nets[0](states).squeeze()

        # Calculate the current logu values (feedforward):
        curr_logu = torch.cat([online_logu(states).squeeze().gather(1, actions.long())
                               for online_logu in self.online_logus], dim=1)

        with torch.no_grad():
            online_logu_next = torch.stack([logu(next_states)
                                            for logu in self.online_logus], dim=0)
            online_curr_logu = torch.stack([logu(states).gather(1, actions)
                                            for logu in self.online_logus], dim=0)

            if self.use_rawlik:
                target_priora = self.target_prior.nets[0](states).squeeze()
                # target_priora /= target_priora.sum(dim=-1, keepdim=True)

                target_prior_next = self.target_prior.nets[0](
                    next_states).squeeze()
                target_prior_next /= target_prior_next.sum(
                    dim=-1, keepdim=True)

            else:
                target_priora = torch.ones(self.batch_size, self.nA, device=self.device) * (1/self.nA)
                target_prior_next = torch.ones(self.batch_size, self.nA, device=self.device) * (1/self.nA)

            log_prior_next = torch.log(target_prior_next + 1e-8)

            # Aggregate the logus:
            online_logu_next = self.aggregator_fn(online_logu_next, dim=0).squeeze(0)
            online_curr_logu = self.aggregator_fn(online_curr_logu, dim=0).squeeze(0)
            online_log_chi = torch.logsumexp(online_logu_next + log_prior_next.repeat(1, 1), dim=-1, keepdim=True)
            # online_curr_logu = online_curr_logu.unsqueeze(-1)

            # TODO: beta missing on the rewards?
            new_theta = -torch.mean( rewards + (online_log_chi - online_curr_logu) / self.beta, dim=0)
            self.theta += self.tau_theta * (new_theta - self.theta)

            target_next_logus = [target_logu(next_states)
                                 for target_logu in self.target_logus]

            # logsumexp over actions:
            target_next_logus = torch.stack(target_next_logus, dim=1)
            target_next_logu = self.aggregator_fn(target_next_logus, dim=1).squeeze(1)
            next_logu = torch.logsumexp(target_next_logu + log_prior_next, dim=-1)

            next_logu = next_logu.reshape(-1, 1)
            assert next_logu.shape == dones.shape
            next_logu = next_logu * (1 - dones)  # + self.theta * dones

            # "Backup" eigenvector equation:
            expected_curr_logu = self.beta * (rewards + self.theta) + next_logu
            expected_curr_logu = expected_curr_logu.squeeze(1)

        # Calculate the logu ("critic") loss:
        loss = 0.5*sum(self.loss_fn(logu, expected_curr_logu)
                       for logu in curr_logu.T)
        
        if self.use_rawlik:
            # prior_loss = self.loss_fn(curr_prior.squeeze(), self.aggregator_fn(online_curr_u,dim=0)[0])
            pistar = torch.exp(self.aggregator_fn(online_curr_logu, dim=0)) * target_priora
            pistar /= pistar.sum(dim=-1, keepdim=True)

            prior_loss = F.kl_div(curr_priora.squeeze().log(
            ), pistar, reduction='batchmean', log_target=False)

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
        self.target_logus.polyak(self.online_logus, self.tau)


    def _update_prior(self):
        # Update the prior:
        if self.use_rawlik:
            self.target_prior.polyak(self.online_prior, self.prior_tau)
import time
import numpy as np
import torch
from BaseAgent import BaseAgent
from Models import LogUNet, OnlineLogUNets, Optimizers, TargetNets
from utils import logger_at_folder

class LogUAgent(BaseAgent):
    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.algo_name = 'LogU'
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
        self.optimizers = Optimizers(opts, self.scheduler_str)

    def exploration_policy(self, state: np.ndarray) -> (int, float):
        # return self.env.action_space.sample(), 0
        kl = 0
        return self.online_logus.choose_action(state, greedy=False), kl

    def evaluation_policy(self, state: np.ndarray) -> int:
        return self.online_logus.choose_action(state, greedy=True)

    def gradient_descent(self, batch, grad_step: int):
        states, actions, next_states, dones, rewards = batch
        # rewards[dones.bool()] -= 10

        # Calculate the current logu values (feedforward):
        curr_logu = torch.cat([online_logu(states).squeeze().gather(1, actions.long())
                               for online_logu in self.online_logus], dim=1)

        with torch.no_grad():
            online_logu_next = torch.stack([logu(next_states)
                                            for logu in self.online_logus], dim=0)
            online_curr_logu = torch.stack([logu(states).gather(1, actions)
                                            for logu in self.online_logus], dim=0)

            # since pi0 is same for all, just do exp(ref_logu) and sum over actions:
            # TODO: should this go outside no grad? Also, is it worth defining a log_prior value?
            online_log_chi = torch.logsumexp(
                online_logu_next, dim=-1) - torch.log(torch.Tensor([self.nA])).to(self.device)
            online_curr_logu = online_curr_logu.squeeze(-1)

            # TODO: beta missing on the rewards?
            self.new_thetas[grad_step, :] = -torch.mean( rewards.squeeze(-1) + (
                online_log_chi - online_curr_logu) / self.beta, dim=1)

            target_next_logus = [target_logu(next_states)
                                 for target_logu in self.target_logus]

            # logsumexp over actions:
            target_next_logus = torch.stack(target_next_logus, dim=1)
            target_next_logu = self.aggregator_fn(target_next_logus, dim=1)
            next_logu = torch.logsumexp(
                target_next_logu, dim=-1) - torch.log(torch.Tensor([self.nA])).to(self.device)
            # next_logu, _ = self.aggregator_fn(next_logus, dim=1)

            next_logu = next_logu.reshape(-1, 1)
            assert next_logu.shape == dones.shape
            next_logu = next_logu  * (1-dones)  # + self.theta * dones

            # "Backup" eigenvector equation:
            expected_curr_logu = self.beta * (rewards + self.theta) + next_logu
            expected_curr_logu = expected_curr_logu.squeeze(1)

        # Calculate the logu ("critic") loss:
        loss = 0.5*sum(self.loss_fn(logu, expected_curr_logu)
                       for logu in curr_logu.T)
        return loss

    def _update_target(self):
        # Do a Polyak update of parameters:
        self.target_logus.polyak(self.online_logus, self.tau)


def main():
    from disc_envs import get_environment
    env_id = get_environment('Pendulum21', nbins=3,
                             max_episode_steps=200, reward_offset=0)

    env_id = 'CartPole-v1'
    # env_id = 'Taxi-v3'
    # env_id = 'CliffWalking-v0'
    # env_id = 'Acrobot-v1'
    # env_id = 'LunarLander-v2'
    # env_id = 'ALE/Pong-v5'
    # env_id = 'PongNoFrameskip-v4'
    # env_id = 'FrozenLake-v1'
    env_id = 'MountainCar-v0'
    # env_id = 'Drug-v0'

    from hparams import cartpole_u as config
    agent = LogUAgent(env_id, **config, device='cpu', log_interval=1500,
                        tensorboard_log='acro', num_nets=2, render=False, aggregator='max',
                        scheduler_str='none')#, beta_schedule = 'linear', beta_end=2.4)
    # Measure the time it takes to learn:
    t0 = time.thread_time_ns()
    agent.learn(total_timesteps=15_000_000)
    t1 = time.thread_time_ns()
    print(f"Time to learn: {(t1-t0)/1e9} seconds")


if __name__ == '__main__':
    for _ in range(1):
        main()

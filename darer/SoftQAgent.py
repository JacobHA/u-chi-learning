import time
import numpy as np
import torch
from BaseAgent import BaseAgent
from Models import SoftQNet, OnlineSoftQNets, Optimizers, TargetNets
from utils import logger_at_folder


class SoftQAgent(BaseAgent):
    def __init__(self,
                 *args,
                 gamma: float = 0.99,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.algo_name = 'SQL'
        self.gamma = gamma
        # Set up the logger:
        self.logger = logger_at_folder(self.tensorboard_log,
                                       algo_name=f'{self.env_str}-{self.algo_name}')
        self.log_hparams(self.logger)
        self._initialize_networks()

    def _initialize_networks(self):
        self.online_softqs = OnlineSoftQNets([SoftQNet(self.env, 
                                                  hidden_dim=self.hidden_dim, 
                                                  device=self.device)
                                              for _ in range(self.num_nets)],
                                            beta=self.beta,
                                            aggregator_fn=self.aggregator_fn)
        # alias for compatibility as self.model:
        self.model = self.online_softqs

        self.target_softqs = TargetNets([SoftQNet(self.env, 
                                                  hidden_dim=self.hidden_dim, 
                                                  device=self.device)
                                        for _ in range(self.num_nets)])
        self.target_softqs.load_state_dicts(
            [softq.state_dict() for softq in self.online_softqs])
        # Make (all) softqs learnable:
        opts = [torch.optim.Adam(softq.parameters(), lr=self.learning_rate)
                for softq in self.online_softqs]
        self.optimizers = Optimizers(opts, self.scheduler_str)

    def exploration_policy(self, state: np.ndarray) -> (int, float):
        # return self.env.action_space.sample()
        kl = 0
        return self.online_softqs.choose_action(state), kl

    def evaluation_policy(self, state: np.ndarray) -> int:
        return self.online_softqs.choose_action(state, greedy=True)

    def gradient_descent(self, batch, grad_step: int):
        states, actions, next_states, dones, rewards = batch

        # Calculate the current softq values (feedforward):
        curr_softq = torch.cat([online_softq(states).squeeze().gather(1, actions.long())
                               for online_softq in self.online_softqs], dim=1)

        with torch.no_grad():
            online_softq_next = torch.stack([softq(next_states)
                                            for softq in self.online_softqs], dim=0)
            online_curr_softq = torch.stack([softq(states).gather(1, actions)
                                            for softq in self.online_softqs], dim=0)

            # since pi0 is same for all, just do exp(ref_softq) and sum over actions:
            # TODO: should this go outside no grad? Also, is it worth defining a log_prior value?
            # TODO: fix pi0 constant
            
            online_curr_softq = online_curr_softq.squeeze(-1)

            target_next_softqs = [target_softq(next_states)
                                 for target_softq in self.target_softqs]

            # logsumexp over actions:
            target_next_softqs = torch.stack(target_next_softqs, dim=1)
            next_vs = 1/self.beta * (torch.logsumexp(
                self.beta * target_next_softqs, dim=-1) - torch.log(torch.Tensor([self.nA])).to(self.device))
            next_v, _ = self.aggregator_fn(next_vs, dim=1)

            next_v = next_v.reshape(-1, 1)
            assert next_v.shape == dones.shape
            next_v = next_v * (1-dones)  # + self.theta * dones

            # "Backup" eigenvector equation:
            expected_curr_softq = rewards + self.gamma * next_v
            expected_curr_softq = expected_curr_softq.squeeze(1)

        # Calculate the softq ("critic") loss:
        loss = 0.5*sum(self.loss_fn(softq, expected_curr_softq)
                       for softq in curr_softq.T)
        return loss

    def _update_target(self):
        # Do a Polyak update of parameters:
        self.target_softqs.polyak(self.online_softqs, self.tau)


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
    # env_id = 'MountainCar-v0'
    # env_id = 'Drug-v0'

    from hparams import cartpole_hparams2 as config
    # drop epislon params:
    try:
        config.pop('exploration_final_eps')
        config.pop('exploration_fraction')
    except:
        pass
    config['beta'] = 1
    config['gamma'] = 0.98

    agent = SoftQAgent(env_id, **config, device='cpu', log_interval=500,
                 tensorboard_log='pong', num_nets=2, render=False, aggregator='min',
                 scheduler_str='none')  # , beta_schedule = 'linear', beta_end=2.4)
    # Measure the time it takes to learn:
    t0 = time.thread_time_ns()
    agent.learn(total_timesteps=100_000)
    t1 = time.thread_time_ns()
    print(f"Time to learn: {(t1-t0)/1e9} seconds")


if __name__ == '__main__':
    for _ in range(1):
        main()

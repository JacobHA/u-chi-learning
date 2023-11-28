import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional as F
import time
from stable_baselines3.common.buffers import ReplayBuffer
import wandb
import sys
sys.path.append("tabular")
sys.path.append("darer")
from Models import LogUNet, OnlineNets, Optimizers, TargetNets
from utils import env_id_to_envs, get_eigvec_values, get_true_eigvec, is_tabular, log_class_vars, logger_at_folder

HPARAM_ATTRS = {
    'beta': 'beta',
    'learning_rate': 'learning_rate',
    'batch_size': 'batch_size',
    'buffer_size': 'buffer_size',
    'target_update_interval': 'target_update_interval',
    'tau': 'tau',
    'theta_update_interval': 'theta_update_interval',
    'hidden_dim': 'hidden_dim',
    'num_nets': 'num_nets',
    'tau_theta': 'tau_theta',
    'gradient_steps': 'gradient_steps',
    'train_freq': 'train_freq',
    'max_grad_norm': 'max_grad_norm',
    'learning_starts': 'learning_starts',
}

LOG_PARAMS = {
    'time/env. steps': 'env_steps',
    'eval/avg_reward': 'avg_eval_rwd',
    'eval/auc': 'eval_auc',
    'time/num. episodes': 'num_episodes',
    'time/fps': 'fps',
    'time/num. updates': '_n_updates',
    'rollout/beta': 'beta',
    'train/lr': 'lr',
}

str_to_aggregator = {'min': torch.min, 
                     'max': torch.max, 
                     'mean': lambda x, dim: (torch.mean(x, dim=dim), None)}

class LogULearner:
    def __init__(self,
                 env_id,
                 beta,
                 learning_rate,
                 batch_size,
                 buffer_size,
                 target_update_interval,
                 tau,
                 theta_update_interval=1,
                 hidden_dim=64,
                 num_nets=2,
                 tau_theta=0.001,
                 gradient_steps=1,
                 train_freq=-1,
                 max_grad_norm=10,
                 learning_starts=1000,
                 loss_fn=None,
                 device='cpu',
                 render=False,
                 log_dir=None,
                 algo_name='logu',
                 log_interval=1000,
                 save_checkpoints=False,
                 use_wandb=False,
                 aggregator='max',
                 scheduler_str='none',    
                 beta_end=None,             
                 ) -> None:
        
        self.env, self.eval_env = env_id_to_envs(env_id, render)
        self.beta = beta
        self.is_tabular = is_tabular(self.env)
        if self.is_tabular:
            # calculate the eigvec:
            self.true_eigvec = get_true_eigvec(self).A.flatten()
            # normalize:
            self.true_eigvec /= np.linalg.norm(self.true_eigvec)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.tau = tau
        self.hidden_dim = hidden_dim
        self.gradient_steps = gradient_steps
        self.device = device
        self.save_checkpoints = save_checkpoints
        self.log_interval = log_interval
        self.tau_theta = tau_theta
        self.theta_update_interval = theta_update_interval
        self.train_freq = train_freq
        self.max_grad_norm = max_grad_norm
        self.num_nets = num_nets
        self.prior = None
        self.learning_starts = learning_starts
        self.use_wandb = use_wandb
        self.aggregator = aggregator
        self.aggregator_fn = str_to_aggregator[aggregator]
        self.avg_eval_rwd = None
        self.fps = None
        self.beta_end = beta_end
        self.scheduler_str = scheduler_str
        
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size,
                                          observation_space=self.env.observation_space,
                                          action_space=self.env.action_space,
                                          n_envs=1,
                                          handle_timeout_termination=True,
                                          device=device)
        assert isinstance(self.env.action_space, gym.spaces.Discrete), \
            "Only discrete action spaces are supported."
        self.nA = self.env.action_space.n

        self.theta = torch.Tensor([0]).to(self.device)
        self.eval_auc = 0
        self.num_episodes = 0

        # Set up the logger:
        self.logger = logger_at_folder(log_dir, algo_name=f'{env_id}-{algo_name}')
        # Log the hparams:
        log_class_vars(self, HPARAM_ATTRS)
        self.logger.dump()

        self._n_updates = 0
        self.env_steps = 0
        self._initialize_networks()
        self.loss_fn = F.smooth_l1_loss if loss_fn is None else loss_fn

    def _initialize_networks(self):
        self.online_logus = OnlineNets([LogUNet(self.env, hidden_dim=self.hidden_dim, device=self.device)
                                                     for _ in range(self.num_nets)],
                                                     aggregator=self.aggregator)
        
        self.target_logus = TargetNets([LogUNet(self.env, hidden_dim=self.hidden_dim, device=self.device)
                                                     for _ in range(self.num_nets)])
        self.target_logus.load_state_dicts([logu.state_dict() for logu in self.online_logus])
        # Make (all) LogUs learnable:
        opts = [torch.optim.Adam(logu.parameters(), lr=self.learning_rate)
                for logu in self.online_logus]
        self.optimizers = Optimizers(opts, self.scheduler_str)

    def train(self):
        # average self.theta over multiple gradient steps
        new_thetas = torch.zeros(self.gradient_steps, self.num_nets).to(self.device)
        for grad_step in range(self.gradient_steps):
            # Sample a batch from the replay buffer:
            batch = self.replay_buffer.sample(self.batch_size)
            states, actions, next_states, dones, rewards = batch
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
                online_log_chi = torch.logsumexp(online_logu_next, dim=-1) - torch.log(torch.Tensor([self.nA])).to(self.device)
                online_curr_logu = online_curr_logu.squeeze(-1)
                
                new_thetas[grad_step, :] = -torch.mean(rewards.squeeze(-1) + (online_log_chi - online_curr_logu) / self.beta, dim=1)

                target_next_logus = [target_logu(next_states)
                                        for target_logu in self.target_logus]
                
                # logsumexp over actions:
                target_next_logus = torch.stack(target_next_logus, dim=1)
                next_logus = torch.logsumexp(target_next_logus, dim=-1) - torch.log(torch.Tensor([self.nA])).to(self.device)
                next_logu, _ = self.aggregator_fn(next_logus, dim=1)
 
                next_logu = next_logu.reshape(-1, 1)
                assert next_logu.shape == dones.shape
                next_logu = next_logu * (1-dones) # + self.theta * dones

                # "Backup" eigenvector equation:
                expected_curr_logu = self.beta * (rewards + self.theta) + next_logu
                expected_curr_logu = expected_curr_logu.squeeze(1)

            # Calculate the logu ("critic") loss:
            loss = 0.5*sum(self.loss_fn(logu, expected_curr_logu) for logu in curr_logu.T)

            self.logger.record("train/loss", loss.item())
            self.optimizers.zero_grad()
            # Increase update counter
            self._n_updates += self.gradient_steps

            # Clip gradient norm
            loss.backward()
            self.online_logus.clip_grad_norm(self.max_grad_norm)
            self.optimizers.step()
        #TODO: Clamp based on reward range
        # new_thetas = torch.clamp(new_thetas, self.min_rwd, self.max_rwd)
        # Log both theta values:
        for idx, new_theta in enumerate(new_thetas.T):
            self.logger.record(f"train/theta_{idx}", new_theta.mean().item())
        new_theta = self.aggregator_fn(new_thetas.mean(dim=0), dim=0)[0]

        # Can't use env_steps b/c we are inside the learn function which is called only
        # every train_freq steps:
        if self._n_updates % self.theta_update_interval == 0:
            self.theta = self.tau_theta * self.theta + (1 - self.tau_theta) * new_theta

        # Log info from this training cycle:
        self.logger.record("train/theta", self.theta.item())
        self.logger.record("train/avg logu", curr_logu.mean().item())
        self.logger.record("train/min logu", curr_logu.min().item())
        self.logger.record("train/max logu", curr_logu.max().item())
        # Log the max gradient:
        total_norm = torch.max(torch.stack(
                    [px.grad.detach().abs().max() 
                        for p in self.online_logus.parameters() for px in p]
                    ))
        self.logger.record("train/max_grad", total_norm.item())


    def learn(self, total_timesteps, beta_schedule=None):
        # Start a timer to log fps:
        self.initial_time = time.thread_time_ns()
        self.betas = self._beta_scheduler(beta_schedule, total_timesteps)
        while self.env_steps < total_timesteps:
            state, _ = self.env.reset()

            episode_reward = 0
            done = False
            self.num_episodes += 1
            self.rollout_reward = 0
            while not done and self.env_steps < total_timesteps:
                # take a random action:
                if self.env_steps < self.learning_starts:
                    action = self.env.action_space.sample()
                else:
                    action = self.online_logus.choose_action(state)
                    # action = self.online_logus.greedy_action(state)
                    # action = self.env.action_space.sample()

                next_state, reward, terminated, truncated, infos = self.env.step(
                    action)
                done = terminated or truncated
                self.rollout_reward += reward

                train_this_step = (self.train_freq == -1 and terminated) or \
                    (self.train_freq != -1 and self.env_steps % self.train_freq == 0)
                if train_this_step:
                    if self.env_steps > self.batch_size:#self.learning_starts:
                        self.train()

                if self.env_steps % self.target_update_interval == 0:
                    # Do a Polyak update of parameters:
                    self.target_logus.polyak(self.online_logus, self.tau)

                self.beta = self.betas[self.env_steps - 1]

                self.env_steps += 1
                episode_reward += reward

                # Add the transition to the replay buffer:
                sarsa = (state, next_state, action, reward, terminated)
                self.replay_buffer.add(*sarsa, [infos])
                state = next_state
                self._log_stats()

            if done:
                self.logger.record("rollout/reward", self.rollout_reward)


    def _log_stats(self):
        if self.env_steps % self.log_interval == 0:
            # end timer:
            t_final = time.thread_time_ns()
            # fps averaged over log_interval steps:
            self.fps = self.log_interval / ((t_final - self.initial_time + 1e-16) / 1e9)

            if self.env_steps >= 0:
                self.avg_eval_rwd = self.evaluate()
                self.eval_auc += self.avg_eval_rwd
            if self.save_checkpoints:
                raise NotImplementedError
                torch.save(self.online_logu.state_dict(),
                           'sql-policy.para')
            # Get the current learning rate from the optimizer:
            self.lr = self.optimizers.get_lr()
            log_class_vars(self, LOG_PARAMS, use_wandb=self.use_wandb)


            if self.is_tabular:
                # Record the error in the eigenvector:
                fa_eigvec = get_eigvec_values(self).flatten()
                # normalize:
                fa_eigvec /= np.linalg.norm(fa_eigvec)
                err = np.abs(self.true_eigvec - fa_eigvec).mean()
                self.logger.record('train/eigvec_err', err.item())

            if self.use_wandb:
                wandb.log({'env_steps': self.env_steps,
                           'eval/avg_reward': self.avg_eval_rwd})
            self.logger.dump(step=self.env_steps)
            self.initial_time = time.thread_time_ns()

    def evaluate(self, n_episodes=5):
        # run the current policy and return the average reward
        self.initial_time = time.process_time_ns()
        avg_reward = 0.
        # log the action frequencies:
        action_freqs = torch.zeros(self.nA)
        n_steps = 0
        for ep in range(n_episodes):
            state, _ = self.eval_env.reset()
            done = False
            while not done:
                action = self.online_logus.greedy_action(state)
                # action = self.online_logus.choose_action(state)
                action_freqs[action] += 1
                action = action.item()
                # action = self.online_logus.choose_action(state)
                n_steps += 1

                next_state, reward, terminated, truncated, info = self.eval_env.step(
                    action)
                avg_reward += reward
                state = next_state
                done = terminated or truncated

        avg_reward /= n_episodes
        # log the action frequencies:
        action_freqs /= n_episodes
        for i, freq in enumerate(action_freqs):
            self.logger.record(f'eval/action_freq_{i}', freq.item())
        final_time = time.process_time_ns()
        eval_time = (final_time - self.initial_time + 1e-12) / 1e9
        eval_fps = n_steps / eval_time
        self.logger.record('eval/time', eval_time)
        self.logger.record('eval/fps', eval_fps)
        self.eval_time = eval_time
        self.eval_fps = eval_fps
        self.avg_eval_rwd = avg_reward
        return avg_reward

    def _beta_scheduler(self, beta_schedule, total_timesteps):
        # setup beta scheduling
        if beta_schedule == 'exp':
            self.betas = torch.exp(torch.linspace(np.log(self.beta), np.log(self.beta_end), total_timesteps)).to(self.device)
        elif beta_schedule == 'linear':
            self.betas = torch.linspace(self.beta, self.beta_end, total_timesteps).to(self.device)
        elif beta_schedule == 'none':
            self.betas = torch.tensor([self.beta] * total_timesteps).to(self.device)
        else:
            raise NotImplementedError(f"Unknown beta schedule: {beta_schedule}")
        return self.betas
    
def main():
    from disc_envs import get_environment

    # env_id = 'CartPole-v1'
    # env_id = 'Taxi-v3'
    # env_id = 'CliffWalking-v0'
    env_id = 'Acrobot-v1'
    # env_id = 'LunarLander-v2'
    # env_id = 'ALE/Pong-v5'
    # env_id = 'FrozenLake-v1'
    # env_id = 'MountainCar-v0'
    # env_id = 'Drug-v0'
    # env_id = get_environment('Pendulum21', nbins=3, max_episode_steps=200, reward_offset=0)

    from hparams import acrobot_logu as config
    agent = LogULearner(env_id, **config, device='cpu', log_interval=1000,
                        log_dir='pend', num_nets=2, render=0, aggregator='max',
                        scheduler_str='none', algo_name='std', beta_end=5)
    # Measure the time it takes to learn:
    t0 = time.thread_time_ns()
    agent.learn(total_timesteps=10_000, beta_schedule='none')
    t1 = time.thread_time_ns()
    print(f"Time to learn: {(t1-t0)/1e9} seconds")

if __name__ == '__main__':
    for _ in range(1):
        main()

from utils import env_id_to_envs, get_eigvec_values, get_true_eigvec, is_tabular, log_class_vars, logger_at_folder
from Models import LogUNet, OnlineNets, Optimizers, TargetNets
import matplotlib.pyplot as plt

def show_frames(frames):
    # Assuming frames is a numpy array of shape (w, h, 4)
    for i in range(frames.shape[2]):
        plt.subplot(2, 2, i+1)
        plt.imshow(frames[:,:,i])
        plt.axis('off')
    plt.show()
import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional as F
import time
# temporarily fix the stable-baselines3 bug:
# from stable_baselines3.common.buffers import ReplayBuffer
from sb3buffers import ReplayBuffer
from sb3buffertensors import ReplayBufferTensors
# from stable_baselines3.common.envs import SubprocVecEnv
import wandb
import sys

sys.path.append("tabular")
sys.path.append("darer")

HPARAM_ATTRS = {
    'beta', 'learning_rate', 'batch_size', 'buffer_size',
    'target_update_interval', 'tau', 'theta_update_interval',
    'hidden_dim', 'num_nets', 'tau_theta', 'gradient_steps',
    'train_freq', 'max_grad_norm', 'learning_starts'
}

LOG_PARAMS = {
    'time/env. steps': 'env_steps',
    'eval/avg_reward': 'avg_eval_rwd',
    'eval/auc': 'eval_auc',
    'time/num. episodes': 'num_episodes',
    'time/fps': 'fps',
    'time/num. updates': '_n_updates',
    'rollout/beta': 'beta',
}

str_to_aggregator = {'min': torch.min, 'max': torch.max, 'mean': torch.mean}


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
                 n_envs=5,
                 tensor_buff=False,
                 frameskip=1,
                 grayscale_obs=False,
                 framestack_k=None,
                 ) -> None:
        self.env, self.eval_env = env_id_to_envs(
            env_id, render, n_envs=n_envs, frameskip=frameskip, framestack_k=framestack_k, grayscale_obs=grayscale_obs)
        self.n_envs = n_envs
        self.is_vector_env = n_envs > 1
        # self.envs = gym.make_vec(env_id, render_mode='human' if render else None, num_envs=8)
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

        RPB = ReplayBufferTensors if tensor_buff else ReplayBuffer
        self.replay_buffer = RPB(buffer_size=buffer_size,
                                 observation_space=self.env.observation_space,
                                 action_space=self.env.action_space,
                                 n_envs=n_envs,
                                 handle_timeout_termination=False,
                                 device=device,
                                 optimize_memory_usage=True)
        self.nA = self.env.action_space.nvec[0] if isinstance(self.env.action_space,
                                                              gym.spaces.MultiDiscrete) else self.env.action_space.n
        self.ref_action = None
        self.ref_state = None
        self.ref_reward = None
        self.theta = torch.Tensor([0]).to(self.device)
        self.eval_auc = 0
        self.num_episodes = 0

        # Set up the logger:
        self.logger = logger_at_folder(
            log_dir, algo_name=f'{env_id}-{algo_name}')
        # Log the hparams:
        for key in HPARAM_ATTRS:
            self.logger.record(f"hparams/{key}", self.__dict__[key])
        self.logger.dump()

        self._n_updates = 0
        self.env_steps = 0
        self._initialize_networks()
        self.loss_fn = F.smooth_l1_loss if loss_fn is None else loss_fn

    def _initialize_networks(self):
        self.online_logus = OnlineNets(
            [LogUNet(self.env, hidden_dim=self.hidden_dim, device=self.device)
             for _ in range(self.num_nets)],
            aggregator=self.aggregator,
            is_vector_env=self.is_vector_env,
        )

        self.target_logus = TargetNets([LogUNet(self.env, hidden_dim=self.hidden_dim, device=self.device)
                                        for _ in range(self.num_nets)])
        self.target_logus.load_state_dicts(
            [logu.state_dict() for logu in self.online_logus])
        # Make (all) LogUs learnable:
        opts = [torch.optim.Adam(logu.parameters(), lr=self.learning_rate)
                for logu in self.online_logus]
        self.optimizers = Optimizers(opts, self.scheduler_str)

    def train(self):
        # average self.theta over multiple gradient steps
        new_thetas = torch.zeros(
            self.gradient_steps, self.num_nets).to(self.device)
        for grad_step in range(self.gradient_steps):
            # Sample a batch from the replay buffer:
            batch = self.replay_buffer.sample(self.batch_size)
            states, actions, next_states, dones, rewards = batch
            # Calculate the current logu values (feedforward):
            curr_logu = torch.cat([online_logu(states).squeeze().gather(1, actions.long())
                                   for online_logu in self.online_logus], dim=1)

            with torch.no_grad():
                # ref_logu_next = torch.stack([logu(self.ref_next_state)
                #             for logu in self.online_logus], dim=0)
                # ref_curr_logu = torch.stack([logu(self.ref_state)[:,self.ref_action]
                #             for logu in self.online_logus], dim=0)

                ref_logu_next = torch.stack([logu(next_states)
                                             for logu in self.online_logus], dim=0)

                # TODO: replace with curr_logu already calculated above:
                ref_curr_logu = torch.stack([logu(states).gather(1, actions.long())
                                             for logu in self.online_logus], dim=0)
                # since pi0 is same for all, just do exp(ref_logu) and sum over actions:

                log_ref_chi = torch.logsumexp(ref_logu_next, dim=-1) - torch.log(torch.Tensor([self.nA])).to(
                    self.device)
                # log_ref_chi = log_ref_chi.unsqueeze(-1)
                ref_curr_logu = ref_curr_logu.squeeze(-1)

                new_thetas[grad_step, :] = torch.mean(
                    -(rewards.squeeze(-1) + (log_ref_chi - ref_curr_logu) / self.beta), dim=1)
                # new_thetas[grad_step, :] = torch.mean(-(self.ref_reward + (log_ref_chi - ref_curr_logu) / self.beta), dim=1)

                target_next_logus = [target_logu(next_states)
                                     for target_logu in self.target_logus]

                self.logger.record("train/target_min_logu",
                                   target_next_logus[0].min().item())
                self.logger.record("train/target_max_logu",
                                   target_next_logus[0].max().item())
                # logsumexp over actions:
                target_next_logus = torch.stack(target_next_logus, dim=1)
                next_logus = torch.logsumexp(target_next_logus, dim=-1) - torch.log(torch.Tensor([self.nA])).to(
                    self.device)
                next_logu = self.aggregator_fn(next_logus, dim=1)
                # handle both mean and min/max aggregators
                next_logu = next_logu if self.aggregator_fn is torch.mean else next_logu[0]

                next_logu = next_logu.reshape(-1, 1)
                assert next_logu.shape == dones.shape
                next_logu = next_logu * (1 - dones)  # + self.theta * dones

                # "Backup" eigenvector equation:
                expected_curr_logu = self.beta * \
                    (rewards + self.theta) + next_logu
                expected_curr_logu = expected_curr_logu.squeeze(1)

            self.logger.record("train/theta", self.theta.item())
            self.logger.record("train/avg logu", curr_logu.mean().item())
            self.logger.record("train/min logu", curr_logu.min().item())
            self.logger.record("train/max logu", curr_logu.max().item())

            # Calculate the logu ("critic") loss:
            loss = 0.5 * sum(self.loss_fn(logu, expected_curr_logu)
                             for logu in curr_logu.T)

            self.logger.record("train/loss", loss.item())
            self.optimizers.zero_grad()
            # Increase update counter
            self._n_updates += self.gradient_steps

            # Clip gradient norm
            loss.backward()
            self.online_logus.clip_grad_norm(self.max_grad_norm)

            # Log the max gradient:
            total_norm = torch.max(torch.stack(
                [px.grad.detach().abs().max()
                 for p in self.online_logus.parameters() for px in p]
            ))
            self.logger.record("train/max_grad", total_norm.item())
            self.optimizers.step()
        # new_thetas = torch.clamp(new_thetas, 1, -1)
        # Log both theta values:
        for idx, new_theta in enumerate(new_thetas.T):
            self.logger.record(f"train/theta_{idx}", new_theta.mean().item())
        new_theta = self.aggregator_fn(new_thetas.mean(dim=0), dim=0)
        new_theta = new_theta if self.aggregator_fn is torch.mean else new_theta[0]

        # Can't use env_steps b/c we are inside the learn function which is called only
        # every train_freq steps:
        if self._n_updates % self.theta_update_interval == 0:
            self.theta = self.tau_theta * self.theta + \
                (1 - self.tau_theta) * new_theta

    def learn(self, total_timesteps, beta_schedule=None):
        # setup beta scheduling
        if beta_schedule == 'exp':
            self.betas = torch.exp(torch.linspace(np.log(self.beta), np.log(self.beta_end), total_timesteps)).to(
                self.device)
        elif beta_schedule == 'linear':
            self.betas = torch.linspace(
                self.beta, self.beta_end, total_timesteps).to(self.device)
        else:
            self.betas = torch.tensor(
                [self.beta] * total_timesteps).to(self.device)

        state, _ = self.env.reset()
        episode_reward = np.zeros(self.n_envs)
        # Start a timer to log fps:
        t0 = time.thread_time_ns()
        while self.env_steps < total_timesteps:
            self.rollout_reward = np.zeros(self.n_envs)
            # take a random action:
            if self.env_steps < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.online_logus.choose_action(state)
                # action = self.online_logus.greedy_action(state)
                # action = self.env.action_space.sample()

            next_state, reward, terminated, truncated, infos = self.env.step(
                action)
            done = np.bitwise_or(terminated, truncated)
            self.num_episodes += np.sum(done)
            self.rollout_reward += reward

            train_this_step = (self.train_freq == -1 and terminated) or \
                              (self.train_freq != -1 and self.env_steps %
                               self.train_freq == 0)

            if train_this_step:
                if self.env_steps > self.batch_size:
                    self.train()

            if self.env_steps % self.target_update_interval == 0:
                # Do a Polyak update of parameters:
                self.target_logus.polyak(self.online_logus, self.tau)

            self.beta = self.betas[self.env_steps - 1]

            self.env_steps += 1
            episode_reward += reward

            # Add the transition to the replay buffer:
            sarsa = (state, next_state, action, reward, terminated)
            # todo: fix stable-baselines bug in adding multiple parallel sarsa
            self.replay_buffer.add(*sarsa, [infos])
            state = next_state
            if any(done) if self.is_vector_env else done:
                self.logger.record(
                    "rollout/reward", np.mean(episode_reward[done == True]))
                # reset the terminated environments:
                episode_reward[done == True] = 0

            if all(done) if self.is_vector_env else done:
                self.env.reset()
            if self.env_steps % self.log_interval == 0:
                # end timer:
                t_final = time.thread_time_ns()
                # log_thread = threading.Thread(target=self._log_stats, args=(t0, t_final,self.env_steps))
                # log_thread.start()
                self._log_stats(t0, t_final, self.env_steps)
                t0 = time.thread_time_ns()

    def _log_stats(self, t0, t_final, env_steps):
        # fps averaged over log_interval steps:
        self.fps = self.log_interval / ((t_final - t0) / 1e9)
        if env_steps >= 0:
            self.avg_eval_rwd = self.evaluate()
            self.eval_auc += self.avg_eval_rwd
        if self.save_checkpoints:
            raise NotImplementedError
            torch.save(self.online_logu.state_dict(),
                       'sql-policy.para')
        for log_key, attribute in LOG_PARAMS.items():
            self.logger.record(log_key, self.__dict__[attribute])
        log_class_vars(self, LOG_PARAMS)
        # Get the current learning rate from the optimizer:
        lr = self.optimizers.get_lr()
        self.logger.record('train/lr', lr)

        if self.is_tabular:
            # Record the error in the eigenvector:
            fa_eigvec = get_eigvec_values(self).flatten()
            # normalize:
            fa_eigvec /= np.linalg.norm(fa_eigvec)
            err = np.abs(self.true_eigvec - fa_eigvec).mean()
            self.logger.record('train/eigvec_err', err.item())

        if self.use_wandb:
            wandb.log({'env_steps': env_steps,
                       'eval/avg_reward': self.avg_eval_rwd})
        self.logger.dump(step=env_steps)

    def evaluate(self, n_episodes=1):
        # run the current policy and return the average reward
        initial_time = time.process_time_ns()
        avg_reward = 0.
        # log the action frequencies:
        action_freqs = torch.zeros(self.nA)
        n_steps = 0
        for ep in range(n_episodes):
            state, _ = self.eval_env.reset()
            done = np.zeros(self.n_envs, dtype=bool)
            while not all(done):
                action = self.online_logus.greedy_action(state)
                # action = self.online_logus.choose_action(state)
                action_freqs[action] += 1
                action = action.item() if not self.is_vector_env else action
                # action = self.online_logus.choose_action(state)
                n_steps += 1
                # ensure there is no pending call to reset:

                next_state, reward, terminated, truncated, info = self.eval_env.step(
                    action)

                avg_reward += reward
                state = next_state
                _done = terminated or truncated if not self.is_vector_env else np.bitwise_or(
                    terminated, truncated)
                done = np.bitwise_or(done, _done)

        # close the video recorder if it is open:
        self.eval_env.close()
        if self.n_envs > 1:
            avg_reward = sum(avg_reward) / self.n_envs
        avg_reward /= n_episodes
        # log the action frequencies:
        action_freqs /= n_episodes
        for i, freq in enumerate(action_freqs):
            self.logger.record(f'eval/action_freq_{i}', freq.item())
        final_time = time.process_time_ns()
        eval_time = (final_time - initial_time) / 1e9
        eval_fps = n_steps / eval_time
        self.logger.record('eval/time', eval_time)
        self.logger.record('eval/fps', eval_fps)
        self.eval_time = eval_time
        self.eval_fps = eval_fps
        self.avg_eval_rwd = avg_reward
        return avg_reward


def main(env_id,
         total_timesteps,
         n_envs,
         log_dir,
         scheduler_str,
         aggregator,
         beta_schedule,
         final_beta_multiplier,
         device,
         **kwargs):
    from disc_envs import get_environment
    # env_id = get_environment('Pendulum5', nbins=3, max_episode_steps=200, reward_offset=0)
    if not kwargs:
        print("Using default hparams")
        from hparams import pong_logu as kwargs
    beta_end = final_beta_multiplier * kwargs['beta']
    assert beta_end > kwargs['beta']
    # I'm not sure why, but there's an extra beta_end coming in from somewhere,
    # so I'm just popping it out of kwargs to be safe
    try:
        kwargs.pop('beta_end')
    except KeyError:
        pass
    agent = LogULearner(env_id, **kwargs, device=device, log_interval=5000,
                        log_dir=log_dir, num_nets=2, render=1, aggregator=aggregator,
                        scheduler_str=scheduler_str, algo_name='std', beta_end=beta_end,
                        n_envs=n_envs, frameskip=4, framestack_k=4, grayscale_obs=True,
                        use_wandb=False
                        )
    # hidden_dim=hidden_dim)
    # Measure the time it takes to learn:
    t0 = time.thread_time_ns()
    agent.learn(total_timesteps=total_timesteps, beta_schedule=beta_schedule)
    # log the final auc:
    wandb.log({'eval/auc': agent.eval_auc})
    t1 = time.thread_time_ns()
    print(f"Time to learn: {(t1 - t0) / 1e9} seconds")


if __name__ == '__main__':
    # import cProfile
    # cProfile.run('main()', sort='cumtime')
    # env_id = 'CartPole-v1'
    # env_id = 'Taxi-v3'
    # env_id = 'CliffWalking-v0'
    # env_id = 'Acrobot-v1'
    # env_id = 'LunarLander-v2'
    # env_id = 'ALE/Boxing-v5'
    # env_id = 'ALE/AirRaid-v5'
    env_id = 'PongNoFrameskip-v4'
    # env_id = 'ALE/Pong-v4'
    # env_id = 'FrozenLake-v1'
    # env_id = 'MountainCar-v0'
    # env_id = 'Drug-v0'
    main(env_id, total_timesteps=10_000_000, log_dir='pend', aggregator='max',
         scheduler_str='none', n_envs=1, beta_schedule='none', device='cuda', final_beta_multiplier=10)

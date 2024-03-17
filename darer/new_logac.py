from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape, get_flattened_obs_dim, preprocess_obs
import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
import time
import wandb
import sys
sys.path.append('darer')
from Models import OnlineNets, Optimizers, TargetNets, LogUsa, GaussianPolicy
from utils import env_id_to_envs, log_class_vars, logger_at_folder
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.torch_layers import MlpExtractor, FlattenExtractor
from stable_baselines3.sac.policies import Actor

torch.backends.cudnn.benchmark = True
# raise warning level for debugger:
import warnings
warnings.filterwarnings("error")
HPARAM_ATTRS = ['beta', 'learning_rate', 'batch_size', 'buffer_size',
                'target_update_interval', 'theta_update_interval', 'tau',
                'actor_learning_rate', 'hidden_dim', 'num_nets', 'tau_theta',
                'learning_starts', 'gradient_steps', 'train_freq', 'max_grad_norm']

str_to_aggregator = {'min': torch.min, 
                     'max': torch.max, 
                     'mean': lambda x, dim: (torch.mean(x, dim=dim), None)}

LOG_PARAMS = {
    'time/env. steps': 'env_steps',
    'eval/avg_reward': 'avg_eval_rwd',
    'eval/auc': 'eval_auc',
    'time/num. episodes': 'num_episodes',
    'time/fps': 'fps',
    'time/num. updates': '_n_updates',
    'rollout/beta': 'beta',
}

class LogUActor:
    def __init__(self,
                 env_id,
                 beta,
                 learning_rate,
                 batch_size,
                 buffer_size,
                 target_update_interval,
                 theta_update_interval,
                 tau,
                 actor_learning_rate=None,
                 hidden_dim=64,
                 num_nets=2,
                 tau_theta=0.001,
                 learning_starts=5000,
                 gradient_steps=1,
                 train_freq=1,
                 max_grad_norm=10,
                 device='cpu',
                 aggregator='max',
                 beta_end=None,
                 log_dir=None,
                 render=False,
                 log_interval=1000,
                 save_checkpoints=False,
                 use_wandb=False,
                 ) -> None:
        self.env, self.eval_env = env_id_to_envs(env_id, render)

        self.env_str = self.env.unwrapped.spec.id if hasattr(self.env.unwrapped.spec, 'id') else self.env.unwrapped.id

        self.n_samples = 1

        self.nA = get_action_dim(self.env.action_space)
        self.nS = get_flattened_obs_dim(self.env.observation_space)
        self.beta = beta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.theta_update_interval = theta_update_interval
        self.tau = tau
        self.hidden_dim = hidden_dim
        self.gradient_steps = gradient_steps
        self.learning_starts = learning_starts
        self.device = device
        self.save_checkpoints = save_checkpoints
        self.log_interval = log_interval
        self.tau_theta = tau_theta
        self.train_freq = train_freq
        self.max_grad_norm = max_grad_norm
        self.num_nets = num_nets
        self.prior = None
        self.use_wandb = use_wandb
        self.actor_learning_rate = actor_learning_rate if \
            actor_learning_rate is not None else learning_rate
        self.beta_end = beta_end if beta_end is not None else beta
        self.aggregator = aggregator
        self.aggregator_fn = str_to_aggregator[aggregator]
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size,
                                          observation_space=self.env.observation_space,
                                          action_space=self.env.action_space,
                                          n_envs=1,
                                          handle_timeout_termination=True,
                                          device=device)
        
        self.theta = torch.Tensor([0]).to(self.device)
        self.avg_eval_rwd = 0
        self.eval_auc = 0
        self.num_episodes = 0

        # Set up the logger:
        self.logger = logger_at_folder(log_dir, algo_name=f'{env_id}')
        # Log the hparams:
        for key in HPARAM_ATTRS:
            self.logger.record(f"hparams/{key}", self.__dict__[key])
        self.logger.dump()

        self._n_updates = 0
        self.env_steps = 0
        self._initialize_networks()


    def _initialize_networks(self):
        self.online_logus = OnlineNets([LogUsa(self.env,
                                               hidden_dim=self.hidden_dim,
                                               device=self.device)
                                        for _ in range(self.num_nets)],
                                        aggregator=self.aggregator)
        self.target_logus = TargetNets([LogUsa(self.env,
                                               hidden_dim=self.hidden_dim,
                                               device=self.device)
                                        for _ in range(self.num_nets)])
        self.target_logus.load_state_dicts(
            [logu.state_dict() for logu in self.online_logus])
        # self.actor = GaussianPolicy(self.hidden_dim, 
        #                             self.env.observation_space, self.env.action_space,
        #                             use_action_bounds=True,
        #                             device=self.device)
        self.actor = Actor(self.env.observation_space, self.env.action_space,
                    [self.hidden_dim, self.hidden_dim],
                    FlattenExtractor(self.env.observation_space),
                    self.nS,
                    )
        # send the actor to device:
        self.actor.to(self.device)
        # TODO: Try a fixed covariance network (no/ignored output)
        # Make (all) LogUs and Actor learnable:
        opts = [torch.optim.Adam(logu.parameters(), lr=self.learning_rate)
                for logu in self.online_logus]
        opts.append(torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_learning_rate))
        self.optimizers = Optimizers(opts)

    def train(self,):
        self.actor.set_training_mode(True)
        # average self.theta over multiple gradient steps
        new_thetas = torch.zeros(
            self.gradient_steps, self.num_nets).to(self.device)
        
        for grad_step in range(self.gradient_steps):
            replay = self.replay_buffer.sample(self.batch_size)
            states, actions, next_states, dones, rewards = replay
            # actor_actions, curr_log_prob, means = self.actor.sample(states)
            actor_actions, curr_log_prob = self.actor.action_log_prob(states)

            curr_logu = torch.stack([online_logu(states, actions)
                                   for online_logu in self.online_logus], dim=-1).squeeze(1)
            with torch.no_grad():
                # sampled_action = torch.Tensor(np.array([self.env.action_space.sample() for 
                                                #   _ in range(self.batch_size)])).to(self.device)
                # sampled_action, log_prob = self.actor.action_log_prob(observations)
                sampled_action = actions#self.actor.action_log_prob(next_states)[0]#.squeeze(1)
                # use same number of samples as the batch size for convenience:
                # sampled_action = torch.Tensor(np.array([self.env.action_space.sample() for 
                #                                _ in range(self.batch_size)])).to(self.device)
                # use single sample for now:
                # sampled_action = torch.Tensor(np.array([self.env.action_space.sample()])).to(self.device)
                # repeat the sampled action for each state in batch:
                # sampled_action = sampled_action.repeat(self.batch_size, 1)

                
                ref_logu_next = torch.stack([logu(next_states, sampled_action)
                            for logu in self.online_logus], dim=0)
                ref_curr_logu = torch.stack([logu(states, actions)
                            for logu in self.online_logus], dim=0)  

                log_next_chi = ref_logu_next# - torch.log(torch.Tensor([self.nA])).to(self.device)
                # log_next_chi = log_next_chi.unsqueeze(-1)
                # new_thetas[grad_step, :] = self.ref_reward - log_ref_chi
                new_thetas[grad_step, :] = torch.mean(-(rewards + (log_next_chi - ref_curr_logu) / self.beta), dim=1).T
                
                # rand_actions = np.array([rand_action for _ in range(self.batch_size)])

                # rand_actions = torch.Tensor(np.array([self.env.action_space.sample() for 
                #                                _ in range(self.batch_size)])).to(self.device)
                                
                target_next_logu = torch.stack([target_logu(next_states, sampled_action)
                                                for target_logu in self.target_logus], dim=-1)

                next_logu = self.aggregator_fn(target_next_logu, dim=-1)
                next_logu = next_logu * (1 - dones) #+ self.theta * dones

                expected_curr_logu = self.beta * (rewards + self.theta) + next_logu
                expected_curr_logu = expected_curr_logu.squeeze(1)

            self.logger.record("train/theta", self.theta.item())
            self.logger.record("train/avg logu", curr_logu.mean().item())
            # Huber loss:
            loss = 0.5*sum(F.smooth_l1_loss(logu, expected_curr_logu)
                           for logu in curr_logu.T)
            # MSE loss:
            actor_curr_logu = torch.stack([online_logu(states, actor_actions)
                                         for online_logu in self.online_logus], dim=-1)

            # actor_loss = 0.5 * \
            #     F.smooth_l1_loss(curr_log_prob, self.aggregator_fn(actor_curr_logu))
            # PPO clips the prioritzed sampling
            # ratio = torch.exp(curr_logu - actor_curr_logu.min(dim=-1)[0] )
            # Clip the ratio:
            # eps=0.2
            # ratio = torch.clamp(ratio, 1-eps, 1+eps)
            # actor_loss = torch.log(ratio)
                
            # actor_loss = F.smooth_l1_loss(curr_log_prob, self.aggregator_fn(actor_curr_logu,dim=-1)[0].squeeze())
            actor_loss = (- self.aggregator_fn(actor_curr_logu)).mean()
            self.logger.record("train/log_prob", curr_log_prob.mean().item())
            self.logger.record("train/loss", loss.item())
            self.logger.record("train/actor_loss", actor_loss.item())
            self.optimizers.zero_grad()
            # Increase update counter
            self._n_updates += self.gradient_steps

            # if self._n_updates % 100 == 0:
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            
            # Clip gradient norm
            loss.backward()
            self.online_logus.clip_grad_norm(self.max_grad_norm)

            # Log the average gradient:
            for idx, logu in enumerate(self.online_logus.nets):
                norm = torch.max(torch.stack(
                [p.grad.detach().abs().max() for p in logu.parameters()]
                ))
                self.logger.record(f"grad/logu{idx}_norm", norm.item())
            # actor_norm = torch.max(torch.stack(
            #     [p.grad.detach().abs().max() for p in self.actor.parameters()]
            # ))
            # self.logger.record("grad/actor_norm", actor_norm.item())
            self.optimizers.step()
        
        # TODO: Take the mean, then aggregate:
        # new_theta = new_theta
        grad_avgd_new_thetas = new_thetas.mean(dim=0) 
        new_theta = self.aggregator_fn(grad_avgd_new_thetas, dim=0)
        # record both thetas:
        for idx, theta in enumerate(grad_avgd_new_thetas):
            self.logger.record(f"train/theta_{idx}", theta.item())
        if self._n_updates % self.theta_update_interval == 0:
            self.theta = self.tau_theta * self.theta + \
                (1 - self.tau_theta) * new_theta

    def learn(self, total_timesteps, beta_schedule='none'):
        # Start a timer to log fps:
        self.t0 = time.thread_time_ns()
        # setup beta scheduling
        self.betas = self._setup_beta_schedule(total_timesteps, beta_schedule)
        while self.env_steps < total_timesteps:
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            self.num_episodes += 1
            self.rollout_reward = 0

            while not done and self.env_steps < total_timesteps:
                self.actor.set_training_mode(False)
                if self.env_steps < self.learning_starts:
                    # take a random action:
                    noisy_action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        # noisy_action, logprob, _ = self.actor.sample(state)
                        # noisy_action, _ = self.actor.predict(state)
                        noisy_action = self.env.action_space.sample()
                        # log the logprob:
                        # self.logger.record("rollout/log_prob", logprob.mean().item())
                        # noisy_action = noisy_action.cpu().numpy()
                next_state, reward, terminated, truncated, infos = self.env.step(
                    noisy_action)
                done = terminated or truncated
                self.rollout_reward += reward

                if (self.train_freq == -1 and done) or (self.train_freq != -1 and self.env_steps % self.train_freq == 0):
                    if self.replay_buffer.size() > self.batch_size:
                        self.train()

                if self.env_steps % self.target_update_interval == 0:
                    # Do a Polyak update of parameters:
                    self.target_logus.polyak(self.online_logus, self.tau)
                    
                self.env_steps += 1
                self.beta = self.betas[self.env_steps]
                episode_reward += reward
                infos = [infos]
                self.replay_buffer.add(
                    state, next_state, noisy_action, reward, terminated, infos)
                state = next_state
                
                self._log_stats()
            if done:
                self.logger.record("rollout/reward", self.rollout_reward)

    def _log_stats(self):
        if self.env_steps % self.log_interval == 0:
        # end timer:
            t_final = time.thread_time_ns()
            # fps averaged over log_interval steps:
            self.fps = self.log_interval / ((t_final - self.t0) / 1e9)

            avg_eval_rwd = self.evaluate()
            self.avg_eval_rwd = avg_eval_rwd
            self.eval_auc += avg_eval_rwd
            if self.save_checkpoints:
                torch.save(self.online_logu.state_dict(),
                            'sql-policy.para')
            log_class_vars(self, LOG_PARAMS)

            # Log network params:
            # for idx, logu in enumerate(self.online_logus.nets):
            #     for name, param in logu.named_parameters():
            #         self.logger.record(f"params/logu_{idx}/{name}",
            #                            param.data.mean().item())
            # for name, param in self.actor.named_parameters():
            #     self.logger.record(f"params/actor/{name}",
            #                        param.data.mean().item())

            self.logger.dump(step=self.env_steps)
            if self.use_wandb:
                wandb.log({'env step': self.env_steps, 'avg_eval_rwd': avg_eval_rwd})
            self.t0 = time.thread_time_ns()

    def _setup_beta_schedule(self, total_timesteps, beta_schedule='none'):
        # setup beta scheduling
        if beta_schedule == 'exp':
            self.betas = torch.exp(torch.linspace(np.log(self.beta), np.log(self.beta_end), total_timesteps)).to(self.device)
        elif beta_schedule == 'linear':
            self.betas = torch.linspace(self.beta, self.beta_end, total_timesteps).to(self.device)
        elif beta_schedule == 'none':
            self.betas = torch.tensor([self.beta] * total_timesteps).to(self.device)
        else:
            raise NotImplementedError("beta_schedule must be one of exp, linear, or none")
        return self.betas

    def evaluate(self, n_episodes=5):
        # run the current policy and return the average reward
        avg_reward = 0.
        for ep in range(n_episodes):
            state, _ = self.eval_env.reset()
            done = False
            while not done:
                self.actor.set_training_mode(False)
                with torch.no_grad():
                    # noisyaction, logprob, action = self.actor.sample(state)  # , deterministic=True)
                    action,_ = self.actor.predict(state , deterministic=True)

                    # action = action.cpu().numpy()
                next_state, reward, terminated, truncated, info = self.eval_env.step(
                    action)
                avg_reward += reward
                state = next_state
                done = terminated or truncated

        avg_reward /= n_episodes
        return avg_reward

    def save_video(self):
        video_env = self.env_id
        gym.wrappers.monitoring.video_recorder.VideoRecorder(video_env, path='video.mp4')
        raise NotImplementedError

def main():
    # env_id = 'LunarLanderContinuous-v2'
    # env_id = 'BipedalWalker-v3'
    # env_id = 'CartPole-v1'
    env_id = 'Pendulum-v1'
    # env_id = 'Hopper-v4'
    # env_id = 'HalfCheetah-v4'
    # env_id = 'Ant-v4'
    # env_id = 'Simple-v0'
    from hparams import pendulum_logu as config
    agent = LogUActor(env_id, **config, device='cuda',
                      num_nets=2, log_dir='pend', 
                      actor_learning_rate=1e-4, 
                      render=0, max_grad_norm=10, log_interval=1000)
    agent.learn(total_timesteps=500_000, beta_schedule='linear')


if __name__ == '__main__':
    for _ in range(3):
        main()

import time
import gym
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecNormalize, VecMonitor
import torch
from torch.nn import functional as F
from Models import OnlineNets, Optimizers, TargetNets
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.buffers import ReplayBuffer
from utils import logger_at_folder
from AtariModels import AtariLogU as LogUNet

# env = DummyVecEnv([lambda: gym.make(env)])

# env = FrameStack(env, n_stack=4)


class LogULearner:
    def __init__(self,
                 env_id,
                 beta,
                 learning_rate,
                 batch_size,
                 buffer_size,
                 target_update_interval,
                 tau,
                 hidden_dim=64,
                 num_nets=2,
                 tau_theta=0.001,
                 gradient_steps=1,
                 train_freq=-1,
                 max_grad_norm=10,
                 device='cpu',
                 log_dir=None,
                 log_interval=1000,
                 save_checkpoints=False,
                 ) -> None:
        # self.env = gym.make(env_id)
        # self.env = AtariWrapper(self.env)
        vec_env = make_atari_env(env_id, n_envs=4, seed=0)
        self.env = VecFrameStack(vec_env, n_stack=4)
        # make another instance for evaluation purposes only:
        self.eval_env = make_atari_env(env_id)
        self.eval_env = AtariWrapper(self.eval_env)
        # self._vec_normalize_env = unwrap_vec_normalize(self.env)
        self.beta = beta
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
        self.train_freq = train_freq
        self.max_grad_norm = max_grad_norm
        self.num_nets = num_nets
        self.prior = None

        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size,
                                        observation_space=self.env.observation_space,
                                        action_space=self.env.action_space,
                                        n_envs=1,
                                        device=device)
        self.ref_action = None
        self.ref_state = None
        self.ref_reward = None
        self.theta = torch.Tensor([0]).to(self.device)
        self.eval_auc = 0
        self.num_episodes = 0

        # Set up the logger:
        self.logger = logger_at_folder(
            log_dir, algo_name=f'LogU{num_nets}nets')

        self._n_updates = 0
        self.env_steps = 0
        self._initialize_networks()

    def _initialize_networks(self):
        self.online_logus = OnlineNets(list_of_nets=[LogUNet(self.env)
                                                     for _ in range(self.num_nets)])
        self.target_logus = TargetNets(list_of_nets=[LogUNet(self.env)
                                                     for _ in range(self.num_nets)])
        self.target_logus.load_state_dict(
            [logu.state_dict() for logu in self.online_logus])
        # Make (all) LogUs learnable:
        opts = [torch.optim.Adam(logu.parameters(), lr=self.learning_rate)
                for logu in self.online_logus]
        self.optimizers = Optimizers(opts)
        

    def train(self,):
        # replay = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
        # average self.theta over multiple gradient steps
        new_thetas = torch.zeros(self.gradient_steps)
        for grad_step in range(self.gradient_steps):
            replay = self.replay_buffer.sample(self.batch_size)
            states, actions, next_states, next_actions, dones, rewards = replay

            curr_logu = torch.cat([online_logu(states).gather(1, actions.long())
                                   for online_logu in self.online_logus], dim=1)
            with torch.no_grad():
                ref_logu = [logu(self.ref_next_state) for logu in self.online_logus]
                # ref_chi = torch.stack([logu.get_chi(
                #     ref_logu_val) for ref_logu_val, logu in zip(ref_logu, self.online_logus)],dim=-1)
                # since pi0 is same for all, just do exp(ref_logu) and sum over actions:
                ref_chi = torch.stack([torch.exp(ref_logu_val).sum(dim=-1)
                                       for ref_logu_val in ref_logu], dim=-1)
                new_theta = self.ref_reward - torch.log(ref_chi)
                new_thetas[grad_step] = torch.min(new_theta,dim=-1)[0]
                target_next_logu = torch.cat([target_logu(next_states).gather(1, next_actions.long())
                                              for target_logu in self.target_logus], dim=1)

                next_logu, _ = torch.min(target_next_logu, dim=1, keepdim=True)

                # tile next_logu to match curr_logu:
                next_logu = next_logu.repeat(1, self.num_nets)

                expected_curr_logu = self.beta * \
                    (rewards + self.theta) + (1 - dones) * next_logu

            self.logger.record("train/theta", self.theta.item())
            self.logger.record("train/avg logu", curr_logu.mean().item())
            # Huber loss:
            loss = F.smooth_l1_loss(curr_logu, expected_curr_logu)
            # MSE loss:
            # loss = F.mse_loss(curr_logu, expected_curr_logu)
            self.logger.record("train/loss", loss.item())
            self.optimizers.zero_grad()
            # Increase update counter
            self._n_updates += self.gradient_steps

            # Clip gradient norm
            loss.backward()
            self.online_logus.clip_grad_norm(self.max_grad_norm)

            # Log the average gradient:
            # TODO: put this in a parallel process somehow or use dot prods?
            # total_norm = torch.max(torch.stack(
            #             [p.grad.detach().abs().max() for p in self.online_logu.parameters()]
            #             ))
            # self.logger.record("max_grad", total_norm.item())
            self.optimizers.step()
        # new_thetas = torch.clamp(new_thetas, 0, -1)

        self.theta = self.tau_theta*self.theta + \
            (1 - self.tau_theta) * torch.mean(new_thetas)

    def learn(self, total_timesteps):
        # Start a timer to log fps:
        t0 = time.thread_time_ns()

        while self.env_steps < total_timesteps:
            state = self.env.reset()
            if self.env_steps == 0:
                self.ref_state = state
            episode_reward = 0
            done = False
            # Random choice:
            action = self.online_logus.choose_action(state)

            self.num_episodes += 1
            self.rollout_reward = 0
            while not done:
                # take a random action:
                # action = self.env.action_space.sample()
                next_state, reward, done, _ = self.env.step(action)
                self.rollout_reward += reward
                if self.env_steps == 0:
                    self.ref_action = action
                    self.ref_reward = reward
                    self.ref_next_state = next_state

                # TODO: Shorten this: (?)
                if (self.train_freq == -1 and done) or (self.train_freq != -1 and self.env_steps % self.train_freq == 0):
                    if self.replay_buffer.size() > self.batch_size:  # or learning_starts?
                        self.train()

                if self.env_steps % self.target_update_interval == 0:
                    # Do a Polyak update of parameters:
                    self.target_logus.polyak(
                        self.online_logus.parameters(), self.tau)

                self.env_steps += 1
                next_action = self.online_logus.choose_action(next_state)

                episode_reward += reward
                self.replay_buffer.add(
                    state, next_state, action, next_action, reward, done)
                state = next_state
                action = next_action
                if self.env_steps % self.log_interval == 0:
                    # end timer:
                    t_final = time.thread_time_ns()
                    # fps averaged over log_interval steps:
                    fps = self.log_interval / ((t_final - t0) / 1e9)

                    avg_eval_rwd = self.evaluate()
                    self.eval_auc += avg_eval_rwd
                    if self.save_checkpoints:
                        torch.save(self.online_logu.state_dict(),
                                   'sql-policy.para')
                    self.logger.record("time/env. steps", self.env_steps)
                    self.logger.record("eval/avg_reward", avg_eval_rwd)
                    self.logger.record("eval/auc", self.eval_auc)
                    self.logger.record("time/num. episodes", self.num_episodes)
                    self.logger.record("time/fps", fps)
                    self.logger.dump(step=self.env_steps)
                    t0 = time.thread_time_ns()
                    self.logger.record("rollout/reward", self.rollout_reward)

    def evaluate(self, n_episodes=5):
        # run the current policy and return the average reward
        avg_reward = 0.
        for ep in range(n_episodes):
            state = self.eval_env.reset()
            done = False
            while not done:
                action = self.online_logus.greedy_action(state)
                # action = self.online_logus.choose_action(state)
                # if ep == 0:
                #     self.env.render()

                next_state, reward, done, _ = self.eval_env.step(action)

                avg_reward += reward
                state = next_state
        avg_reward /= n_episodes
        self.eval_env.close()
        return avg_reward
    
def main():
    from hparams import cartpole_hparams0 as config
    env_id = 'PongNoFrameskip-v4'

    agent = LogULearner(env_id, **config, device='cuda', num_nets=1)
    agent.learn(total_timesteps=50_000)


if __name__ == '__main__':
    for _ in range(1):
        main()

import sys
import time
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps

from utils import logger_at_folder

class CustomDQN(DQN):
    def __init__(self, 
                 env_id, 
                 log_interval=500, 
                 hidden_dim=64, 
                 tensorboard_log='', 
                 max_eval_steps=None, 
                 name_suffix='', 
                 **kwargs):
        
        # strip the render arg:
        self.render = kwargs.pop('render', False)
        policy_kwargs = {'net_arch': [hidden_dim, hidden_dim]}
        super().__init__('MlpPolicy', env_id, verbose=4, policy_kwargs=policy_kwargs, **kwargs)
        self.eval_auc = 0
        self.eval_rwd = 0
        self.eval_interval = log_interval
        if max_eval_steps is None:
            self.eval_env = self.env
        else:
            self.eval_env = gym.make(env_id, max_episode_steps=max_eval_steps)
        self.step_to_avg_eval_rwd = {}

        name_suffix = '-' + name_suffix if name_suffix else ''
        self.our_logger = logger_at_folder(tensorboard_log, 
                                           algo_name='DQN'+name_suffix)

    def _on_step(self) -> None:
        # Evaluate the agent and log it if step % log_interval == 0:
        if self._n_calls % self.eval_interval == 0:
            self.eval_rwd = self.evaluate_agent(5)
            self.eval_auc += self.eval_rwd
            self.step_to_avg_eval_rwd[self.num_timesteps] = self.eval_rwd
            self.our_logger.record("eval/auc", self.eval_auc)
            self.our_logger.record("eval/avg_reward", self.eval_rwd)
            # self._dump_logs()#step=self.num_timesteps)
            self.our_logger.dump(step=self.num_timesteps)

        # Do super's self._on_step:
        super()._on_step()

    def evaluate_agent(self, n_episodes=1):
        # Run the current policy and return the average reward
        avg_reward = 0.
        for _ in range(n_episodes):
            state = self.eval_env.reset()
            done = False
            while not done:
                action = self.predict(state, deterministic=True)[0]
                next_state, reward, done, _ = self.eval_env.step(action)
                avg_reward += reward.item()
                state = next_state
        avg_reward /= n_episodes
        self.eval_env.close()
        return float(avg_reward)

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        self.our_logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.our_logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.our_logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
            self.our_logger.record("eval/auc", self.eval_auc)
            self.our_logger.record("eval/avg_reward", self.eval_rwd)

        self.our_logger.record("time/fps", fps)
        self.our_logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.our_logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")

        # Ensure eval/avg_reward is recorded, even if no episode was completed:
        # self.eval_rwd = self.evaluate_agent()
        # self.eval_auc += self.eval_rwd


        if self.use_sde:
            self.our_logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.our_logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        
        # Pass the number of timesteps for tensorboard
        # self.our_logger.dump(step=self.num_timesteps)

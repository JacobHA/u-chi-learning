from stable_baselines3 import PPO
import gymnasium as gym
from utils import logger_at_folder


class CustomPPO(PPO):
    def __init__(self, *args, log_interval=1000, render=0, hidden_dim=64, log_dir=None, **kwargs):
        super().__init__('MlpPolicy', *args, verbose=4, **kwargs)
        self.eval_auc = 0
        self.eval_rwd = 0
        self.eval_interval = log_interval
        self.eval_env = gym.make('HalfCheetah-v4', render_mode='human' if render else None)

        # Translate hidden dim to policy_kwargs:
        self.policy_kwargs = {'net_arch': [hidden_dim, hidden_dim]}

        # Set up logging:
        # self.logger = logger_at_folder(log_dir, 'ppo')
        self.tensorboard_log = log_dir

    def train(self):
        if self.num_timesteps % (self.eval_interval//5) == 0:
            self.eval_rwd = self.evaluate_agent(5)
            self.eval_auc += self.eval_rwd
            self.logger.record("eval/auc", self.eval_auc)
            self.logger.record("eval/avg_reward", self.eval_rwd)
            self.logger.dump(step=self.num_timesteps)
        super().train()
        # self.logger.dump(step=self.num_timesteps)

    def evaluate_agent(self, n_episodes=1):
        # Run the current policy and return the average reward
        avg_reward = 0.
        for _ in range(n_episodes):
            state, _ = self.eval_env.reset()
            done = False
            while not done:
                action = self.predict(state, deterministic=True)[0]
                next_state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                avg_reward += reward
                state = next_state
        avg_reward /= n_episodes
        #self.eval_env.close()
        return float(avg_reward)


if __name__ == '__main__':
    agent = CustomPPO('HalfCheetah-v4', 
                      log_dir='logs', 
                      hidden_dim=128, 
                      n_steps=128, 
                      n_epochs=4, 
                      ent_coef=0.0, 
                      gae_lambda=0.95, 
                      learning_rate=0.0003, 
                      batch_size=256, 
                      render=1)
    agent.learn(total_timesteps=1_000_000)
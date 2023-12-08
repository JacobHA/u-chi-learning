from stable_baselines3 import DQN
from utils import env_id_to_envs

env_id = 'ALE/Pong-v5'
env, eval_env = env_id_to_envs(env_id,
                    render=False,
                    n_envs=1,
                    frameskip=4,
                    framestack_k=4,
                    grayscale_obs=True,
)

model = DQN('CnnPolicy', env, verbose=4, device='cuda')

model.learn(total_timesteps=1000000, log_interval=10)
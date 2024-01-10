import json
import sys
from stable_baselines3 import DQN
sys.path.append("darer")
from utils import env_id_to_envs

env_id = 'PongNoFrameskip-v4'
env, eval_env = env_id_to_envs(env_id, render=False, is_atari=True)


with open("experiments/baselines/dqn-mod.json", "r") as f:
    hparams = json.load(f)

total_steps = 1_200_000
# rename to match stable baselines3:
hparams['gamma'] = hparams.pop('discount_factor')
hparams['exploration_fraction'] = hparams.pop('exploration_final_eps_frame') / total_steps
n_stack = hparams.pop("action_history_len")
# remove unused hparams
momentum = hparams.pop('gradient_momentum')
hparams.pop('squared_gradient_momentum')
hparams.pop('min_squared_gradient')
train_ferq = hparams.pop("train_freq") // hparams.pop('action_repeat')
# optimizer = torch.optim.RMSprop
model = DQN('CnnPolicy', env, verbose=1, device='cuda',
            policy_kwargs={
                'normalize_images': False,
                # 'optimizer_class': optimizer,
                # 'optimizer_kwargs': {"momentum": momentum},
            }, **hparams)

model.learn(total_timesteps=total_steps, log_interval=3)
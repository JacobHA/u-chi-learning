# We run a scaling experiment on acrobot to show how hidden_dim affects performance:
import sys
import os

sys.path.append('darer')
from CustomDQN import CustomDQN
from CustomPPO import CustomPPO
from UAgent import UAgent
from LogUAgent import LogUAgent
from hparams import *

# env_id = 'Acrobot-v1'
env_id = 'CartPole-v1'
algo_str = 'dqn'
filename = f"experiments/scaling_expt/results/{env_id}-{algo_str}.csv"
# Label the columns hidden dim, and the remaining are timesteps (51):
os.makedirs(os.path.dirname(filename), exist_ok=True)
# file = open(filename, 'w+')
# file.write('hidden_dim,')
# for i in range(51):
#     file.write(f'Step {i},')
# file.write('\n')
# file.close()

str_to_algo = {
    'u': UAgent,
    'logu': LogUAgent,
    'ppo': CustomPPO,
    'dqn': CustomDQN
}
algo = str_to_algo[algo_str]
hparams = id_to_hparam_dicts[env_id][algo_str]
def runner(hidden_dim, device, total_timesteps):
    # hparams.pop('hidden_dim')

    hparams['hidden_dim'] = hidden_dim
    agent = algo(env_id, **hparams, log_interval=1000)#, use_wandb=False,
                # device=device, render=0)
    agent.learn(total_timesteps=total_timesteps)
    return agent.step_to_avg_eval_rwd


def main(hidden_dim):
    print(f"Training with hidden_dim: {hidden_dim}")
    reward_dict = runner(hidden_dim, device='cpu', total_timesteps=50_000)
    # Add new line to file:

    with open(filename, 'a+') as f:
        # Add hidden dim row with reward_dict values:
        f.write(f"\n{hidden_dim},")
        for k,v in reward_dict.items():
            f.write(f"{v},")
        f.write('\n')
        

if __name__ == '__main__':
    import multiprocessing as mp
    # multiprocess the hidden dims:
    hidden_dims = [2**i for i in range(1, 11)]
    pool = mp.Pool()
    pool.map(main, hidden_dims)
    pool.close()
    pool.join()
    
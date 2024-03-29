import sys
sys.path.append('../darer')
sys.path.append('darer')
from utils import atari_env_id_to_envs

def test_atari_env_id_to_envs():
    env_id = 'PongNoFrameskip-v4'
    env, eval_env = atari_env_id_to_envs(
        env_id, render=False, n_envs=1, frameskip=4, framestack_k=4)
    obs, _ = env.reset()
    assert obs.shape == (84, 84, 4)
    assert obs.dtype == 'uint8'
    eval_obs, _ = eval_env.reset()
    assert eval_obs.shape == (84, 84, 4)
    assert eval_obs.dtype == 'uint8'

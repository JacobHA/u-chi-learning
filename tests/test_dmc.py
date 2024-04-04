import gymnasium as gym
from dm_control import suite

def test_dmc():
    for domain_name, task_name in suite.BENCHMARKING:
        env_name = f"dm_control/{domain_name}-{task_name}-v0"
        env = gym.make(env_name)

        env.reset()
        env.close()
        
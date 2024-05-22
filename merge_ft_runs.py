import copy
import os


dirs = [
    'vast_8_3070_ft_logs',
    'vast_8_3080_ft_logs',
    'vast_12_3090_ft_logs',
    'vast_2_4090_ft_logs',
    'sjsu_ft_logs'
]
exp = "EVAL"
combined_env_logs = {}
target_dir = "merged_ft_logs"

for dir in dirs:
    envs = os.listdir(f"{dir}/ft_logs/{exp}")
    for env in envs:
        if env not in combined_env_logs:
            combined_env_logs[env] = []
        runs = os.listdir(f"{dir}/ft_logs/{exp}/{env}")
        for run in runs:
            unique_name = copy.deepcopy(run)
            while unique_name in combined_env_logs[env]:
                unique_name += "_"
            combined_env_logs[env].append(unique_name)
            if not os.path.exists(f"{target_dir}/{exp}/{env}"):
                os.makedirs(f"{target_dir}/{exp}/{env}")
            # save the run
            os.system(f"cp -r {dir}/ft_logs/{exp}/{env}/{run} {target_dir}/{exp}/{env}/{unique_name}")

from matplotlib import pyplot as plt
import seaborn as sns
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
import numpy as np
import os
from glob import glob
import pandas as pd
from tbparse import SummaryReader

plot_metrics = ['eval/avg_reward']
env = 'HalfCheetah-v4'
# env='Ant-v4'
algorithms = [f'{env}-ASAC', 'SAC0.990.2', 'SAC', f'{env}-arDDPG']
algorithms += [f'{env}-arSAC-newh', 'SAC0.990.2', 'SAC', f'{env}-arDDPG']
algorithms = list(set(algorithms))

folder = f'ft_logs/EVAL/{env}/'

score_dict = dict([(algo, []) for algo in algorithms])
# temporal_score_dict = dict([(algo, []) for algo in algorithms])
# Load the scores from the files
for subfolder in os.listdir(folder):
    subfolder = os.path.join(folder, subfolder)
    if not os.path.isdir(subfolder) or subfolder.endswith('.png'):
        continue

    algo_name = os.path.basename(subfolder).split('_')[0]
    if algo_name in algorithms:
        # Replace algo_name with the matching value in the lsit:
        # algo_name = [algo for algo in algorithms if algo in algo_name][0]

        log_files = glob(os.path.join(subfolder, '*.tfevents.*'))
        if not log_files:
            print(f"No log files found in {subfolder}")
            continue
        
        # Require only one log file per folder:
        # assert len(log_files) == 1
        print("Processing", os.path.basename(subfolder))

        try:
            for log_file in log_files:
                reader = SummaryReader(log_file)
                df = reader.scalars
                df = df[df['tag'].isin(plot_metrics)]
                # Add a new column with the algo name:
                df['algo'] = algo_name
                # Add run number:
                df['run'] = os.path.basename(subfolder).split('_')[1]
                # algo_data = pd.concat([algo_data, df])
                score_dict[algo_name].append(sum(df['value'].tolist()) * df['step'].values[0] / 1e6)
                # Get rewards at intervals of 5000:
                # reward_sequence = np.array(df['value'][df['step'] % 5000 == 0].tolist())
                # if len(reward_sequence) != 200:
                #     print(f"Warning: {algo_name} has {len(reward_sequence)} frames. Skipping...")
                #     continue

                # temporal_score_dict[algo_name].append(reward_sequence)
        except Exception as e:
            print(f"Error processing: {e}", log_file)
            continue

# Check if no runs were found:
for algo in algorithms:
    if len(score_dict[algo]) == 0:
        print(f"No runs found for {algo} in {env}.")

        # Remove it
        del score_dict[algo]

# Properly shape the score_dict as n_runs by n_tasks (1):
algorithms = list(score_dict.keys())
for algo in algorithms:
    score_dict[algo] = np.array(score_dict[algo]).reshape(-1, 1)

aggregate_func = lambda x: np.array([
  metrics.aggregate_median(x),
  metrics.aggregate_iqm(x),
  metrics.aggregate_mean(x),
  # metrics.aggregate_optimality_gap(x)
  ])
aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
  score_dict, aggregate_func, reps=2000)
fig, axes = plot_utils.plot_interval_estimates(
  aggregate_scores, aggregate_score_cis,
  metric_names=['Median', 'IQM', 'Mean'],
  algorithms=algorithms, xlabel='Average Evaluation Reward',
  xlabel_y_coordinate=-1)

fig.savefig(f'ft_logs/EVAL/{env}/rliabl.png', bbox_inches='tight')

max_t = 1.25 * np.max([np.max(scores) for scores in aggregate_scores.values()])
thresholds = np.linspace(0.0, max_t, 81)
score_distributions, score_distributions_cis = rly.create_performance_profile(
    score_dict, thresholds, reps=2000)
# Plot score distributions
fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
plot_utils.plot_performance_profiles(
  score_distributions, thresholds,
  performance_profile_cis=score_distributions_cis,
  colors=dict(zip(algorithms, sns.color_palette('colorblind'))),
  xlabel=r'Human Normalized Score $(\tau)$',
  ax=ax)

fig.savefig(f'ft_logs/EVAL/{env}/perf_profile.png', bbox_inches='tight')

# Reshape the temporal_score_dict to have shape n_runs by n_tasks by n_frames:
# temporal_score_dict = {algo: np.array(score).reshape(-1,1,-1) for algo, score in temporal_score_dict.items()}
# frames = np.array([1, 10, 25, 50, 75, 100, 125, 150, 175, 200]) - 1
# # frames = np.arange(1,2000, step=100)-1
# frames = np.arange(1, 200, step=10) - 1

# frames_score_dict = {}
# for algo, scores in temporal_score_dict.items():
#     scores = np.array(scores).T
#     if len(scores) != 200:
#         print(f"Warning: {algo} has {len(scores)} scores. Skipping...")
#         continue
#     frames_score_dict[algo] = np.array([scores[frame]
#                                         for frame in frames])

# iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[:,frame])
#                                for frame in range(scores.shape[-1])])
# iqm_scores, iqm_cis = rly.get_interval_estimates(
#   frames_score_dict, iqm, reps=2000)
# plot_utils.plot_sample_efficiency_curve(
#     frames+1, iqm_scores, iqm_cis, algorithms=algorithms,
#     xlabel=r'Number of Frames (in millions)',
#     ylabel='IQM Human Normalized Score')
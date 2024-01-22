lunar_ppo = {
    'batch_size': 64,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'gae_lambda': 0.98,
    'gamma': 0.999,
    # 'learning_rate': 0.001,
    'n_epochs': 4,
    'n_steps': 32,
    'hidden_dim': 256,
}

lunar_dqn = {
    'batch_size': 128,
    'buffer_size': 50_000,
    'exploration_final_eps': 0.1,
    'exploration_fraction': 0.12,
    'gamma': 0.99,
    'gradient_steps': -1,
    'learning_rate': 0.00063,
    'learning_starts': 0,
    'hidden_dim': 256,
    'target_update_interval': 250,
    'train_freq': 4,
}

cartpole_hparams2 = {
    'batch_size': 860,
    'beta': 10,
    'buffer_size': 100_000,
    'gradient_steps': 1,
    'learning_rate': 1e-3,
    'target_update_interval': 300,
    'tau': 0.64,
    'tau_theta': 0.99,
    'theta_update_interval': 8,
    'hidden_dim': 64,
    'train_freq': 1,
    'learning_starts': 1000
}

cartpole_hparams3 = {
    'batch_size': 32,
    'beta': 0.1,
    # 'beta_schedule': 'linear',
    # 'beta_end': 3,
    'buffer_size': 500_000,
    'gradient_steps': 1,
    'learning_rate': 2.5e-4,
    'target_update_interval': 10_000,
    'tau': 0.99,
    'tau_theta': 0.85,
    'theta_update_interval': 1_000,
    'hidden_dim': 256,
    'train_freq': 4,
    'learning_starts': 50_000
}

cartpole_rawlik = {
    'batch_size': 16,
    'beta': 0.8,
    # 'beta_schedule': 'linear',
    'buffer_size': 100_000,
    # 'final_beta_multiplier': 6,
    'hidden_dim': 64,
    'learning_rate': 0.009,
    'learning_starts': 1000,
    'prior_tau': 0.35,
    'prior_update_interval': 64,
    'target_update_interval': 200,
    'tau': 0.3,
    'tau_theta': 0.6,
    'theta_update_interval': 100,
    'train_freq': 3,
    'learning_starts': 500,
}

mcar_hparams = {
    'beta': 1.7/10,
    'batch_size': 64,
    'buffer_size': 500_000,
    'gradient_steps': 6,
    'learning_rate': 2e-4,
    'target_update_interval': 2300,
    'tau': 0.3,
    'tau_theta': 0.67,
    'theta_update_interval': 1000,
    'hidden_dim': 64,
    'train_freq': 6,
    'learning_starts': 0.05*500_000,
    'aggregator': 'max'
}


lunar_logu = {
    'beta': 10/40,
    'batch_size': 650,
    'buffer_size': 100_000,
    'gradient_steps': 1,
    'learning_rate': 5.5e-4,
    'target_update_interval': 2000,
    'theta_update_interval': 1,
    'tau': 0.25,
    'tau_theta': 0.7,
    'hidden_dim': 256,
    'train_freq': 1,
    'learning_starts': 5_000
}

lunar_u = {
    'beta': 10/40,
    'batch_size': 512,
    'buffer_size': 100_000,
    'gradient_steps': 1,
    'learning_rate': 5.5e-4,
    'target_update_interval': 2000,
    'theta_update_interval': 1,
    'tau': 0.25,
    'tau_theta': 0.7,
    'hidden_dim': 256,
    'train_freq': 1,
    'learning_starts': 5_000
}

sac_hparams2 = {
    'beta': 80,
    'batch_size': 32,
    'buffer_size': 1_000_000,
    'gradient_steps': 1,
    'learning_rate': 3e-4,
    'target_update_interval': 1,
    'tau': 0.005,
    'tau_theta': 0.995,
    'hidden_dim': 64,
    'train_freq': 1
}
cartpole_ppo = {
    'batch_size': 256,
    'clip_range': 0.2,
    'ent_coef': 0.0,
    'gae_lambda': 0.8,
    'gamma': 0.98,
    'learning_rate': 0.001,
    'n_epochs': 20,
    'n_steps': 32,
    'hidden_dim': 64,
}

acrobot_ppo = {
    'ent_coef': 0,
    'gae_lambda': 0.94,
    'n_epochs': 4,
    'n_steps': 256,
    # 'normalize': True,
    # 'normalize_kwargs': {'norm_obs': True, 'norm_reward': False}
}

cartpole_hparams1 = {
    'beta': 1.0,
    'batch_size': 256,
    'buffer_size': 100_000,
    'gradient_steps': 1,
    'learning_rate': 3.6e-3,
    'target_update_interval': 125,
    'tau': 0.7,
    'tau_theta': 0.7,
    'train_freq': 4,
    'hidden_dim': 512,
}

cartpole_dqn = {
    'batch_size': 64,
    'buffer_size': 100000,
    'exploration_final_eps': 0.04,
    'exploration_fraction': 0.16,
    'gamma': 0.99,
    'gradient_steps': 128,
    'hidden_dim': 256,
    'learning_rate': 0.0023,
    'learning_starts': 1000,
    'target_update_interval': 10,
    'tau': 1.0,
    'train_freq': 256,
}

pong_logu = {
    'beta': 0.07,
    'batch_size': 256,
    'buffer_size': 50_000,
    'gradient_steps': 1,
    'learning_rate': 4.e-4,
    'target_update_interval': 3500,
    'tau': 0.35,
    'tau_theta': 0.97,
    'train_freq': 4,
    'learning_starts': 80_000,
    'theta_update_interval': 125,
}

pong_logu0 = {
    'beta': 0.2,
    'batch_size': 32,
    'buffer_size': 50_000,
    'gradient_steps': 1,
    'learning_rate': 1.e-3,
    'target_update_interval': 10000,
    'tau': 0.01,
    'tau_theta': 0.98,
    'train_freq': 4,
    'learning_starts': 15_000,
    'theta_update_interval': 100,
}

nature_pong = {
  "batch_size": 128,
  "beta": 0.1,
  "buffer_size": 300_000,
  "tau": 0.9,
  "train_freq": 4,
  "learning_starts": 50000 ,
  "learning_rate": 0.001,#00025 ,
#   "gradient_momentum": 0.95 ,
#   "squared_gradient_momentum": 0.95 ,
#   "min_squared_gradient": 0.01 ,
#   "action_history_len": 4 ,
#   "action_repeat": 4 ,
#   "discount_factor": 0.99 ,
  "target_update_interval": 20000,
  'aggregator': 'max',
  'hidden_dim': 512,
}

cartpole_u = {
    'beta': 0.01,
    'batch_size': 32,#1150,
    'buffer_size': 100_000,
    'gradient_steps': 4,
    'learning_rate': 3e-4,
    'target_update_interval': 1500,
    'learning_starts': 10000,
    'tau': 0.04,
    'tau_theta': 0.15,
    'theta_update_interval': 875,
    'train_freq': 10,
    'hidden_dim': 128,
    'aggregator': 'max'
}

cartpole_u2 = {
    'batch_size': 128,
    'beta': 0.7,
    'buffer_size': 50_000,
    'hidden_dim': 64,
    'learning_rate': 0.03,
    'learning_starts': 0.005*50_000,
    'target_update_interval': 100,
    'tau': 0.65,
    'tau_theta': 0.71,
    'theta_update_interval': 415,#750,
    'train_freq': 1,
    'gradient_steps': 1,
    'aggregator': 'min'
}
maze = {
    'batch_size': 256,
    'beta': 0.5,
    'buffer_size': 100_000,
    'hidden_dim': 64,
    'learning_rate': 8e-4,
    'learning_starts': 0.2*50_000,
    'target_update_interval': 500,
    'tau': 0.9,
    'tau_theta': 0.9,
    'theta_update_interval': 100,#750,
    'train_freq': 1,
    'aggregator': 'max'
}

acrobot_logu = {
    'beta': 0.35/7,
    'batch_size': 1200//2,
    'buffer_size': 10_000*3,
    'gradient_steps': 1,
    'learning_rate': 1.e-3/3,
    'target_update_interval': 40*4,
    'tau': 0.4,
    'tau_theta': 0.95,
    'train_freq': 1,
    'hidden_dim': 64,
    'theta_update_interval': 4*2,
    'learning_starts': 1_000
}

acrobot_u = {
    'beta': 1.4,
    'batch_size': 32,
    'buffer_size': 50_000,
    'gradient_steps': 7,
    'learning_rate': 1.5e-3,
    'learning_starts': 0,
    'target_update_interval': 100,
    'tau': 0.78,
    'tau_theta': 0.7,
    'train_freq': 7,
    'hidden_dim': 64,
    'theta_update_interval': 6,
    'aggregator': 'max'
}

pendulum_logu = {
    'aggregator': 'max',
    'batch_size': 64,#0,
    'beta': 0.4,
    'beta_schedule': 'linear',
    'buffer_size': 100_000,
    # 'final_beta_multiplier': 6,
    'beta_end': 4.4,
    'gradient_steps': 1,
    'hidden_dim': 256,
    'learning_rate': 5e-4,
    'learning_starts': 15_000,
    'target_update_interval': 325,
    'tau': 0.3,
    'tau_theta': 0.96,
    'theta_update_interval': 500,
    'train_freq': 4,
}

cheetah_hparams = {
    'batch_size': 500,
    'beta': 6.1,
    'buffer_size': 100_000,
    'gradient_steps': 50,
    'learning_rate': 3e-4,
    'target_update_interval': 1000,
    'tau': 0.3,
    'tau_theta': 0.8,
    'train_freq': 100,
    'hidden_dim': 128,
    'learning_starts': 5000
}


cheetah_hparams2 = {
    'batch_size': 600,
    'beta': 0.5,
    'buffer_size': 1_000_000,
    'gradient_steps': 10,
    'learning_rate': 3e-4,
    'target_update_interval': 650,
    'tau': 0.95,
    'tau_theta': 0.98,
    'train_freq': 20,
    'hidden_dim': 128,
    'learning_starts': 15_000
}

acrobot_dqn = {
    'batch_size': 128,
    'buffer_size': 50_000,
    'exploration_final_eps': 0.1,
    'exploration_fraction': 0.12,
    'gamma': 0.99,
    'gradient_steps': -1,
    'learning_rate': 0.00063,
    'learning_starts': 0,
    'hidden_dim': 256,
    'target_update_interval': 250,
    'train_freq': 4,
}

mcar_dqn = {
    'batch_size': 128,
    'buffer_size': 10_000,
    'exploration_final_eps': 0.07,
    'exploration_fraction': 0.2,
    'gamma': 0.98,
    'gradient_steps': 8,
    'hidden_dim': 256,
    'learning_rate': 0.004,
    'learning_starts': 1000,
    'target_update_interval': 600,
    'train_freq': 16,

}

mcar_u = {
    'batch_size': 128,
    'beta': 1.05,
    'buffer_size': 100_000,
    'gradient_steps': 8,
    'learning_rate': 0.02,
    'learning_starts': 0.12*100_000,
    'target_update_interval': 100,
    'tau': 0.95,
    'tau_theta': 0.75,
    'theta_update_interval': 50,#750,
    'train_freq': 8,
    'hidden_dim': 32,
    'aggregator': 'max'
}
# Set up a table of algos/envs to configs:
cartpoles = {
    'u': cartpole_u2,
    'logu': cartpole_hparams2,
    'dqn': cartpole_dqn,
    'ppo': cartpole_ppo
}

acrobots = {
    'logu': acrobot_logu,
    'u': acrobot_u,
    'ppo': acrobot_ppo,
    'dqn': acrobot_dqn
}

mcars = {
    'logu': mcar_hparams,
    # 'ppo': mcar_ppo,
    'dqn': mcar_dqn,
}

lunars = {
    'logu': lunar_logu,
    'u': lunar_u,
    'ppo': lunar_ppo,
    'dqn': lunar_dqn
}

id_to_hparam_dicts = {
    'CartPole-v1': cartpoles,
    'Acrobot-v1': acrobots,
    'MountainCar-v0': mcars,
    'LunarLander-v2': lunars,
}
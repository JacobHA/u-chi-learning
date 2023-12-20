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
    'learning_starts': 10000
}
mcar_hparams = {
    'beta': 0.078,
    'batch_size': 950,
    'buffer_size': 50000,
    'gradient_steps': 25,
    'learning_rate': 5e-4,
    'target_update_interval': 270,
    'tau': 0.28,
    'tau_theta': 0.95,
    'hidden_dim': 128,
    'train_freq': 50,
    'learning_starts': 25000
}

lunar_logu = {
    'beta': 10,
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
    'exploration_fraction': 0.12,
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
    'batch_size': 64,
    'buffer_size': 30_000,
    'gradient_steps': 1,
    'learning_rate': 1.e-3,
    'target_update_interval': 10000,
    'tau': 0.01,
    'tau_theta': 0.98,
    'train_freq': 4,
    'learning_starts': 15_000,
    'theta_update_interval': 100,
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
    'learning_starts': 1_000*5
}

pendulum_logu = {
    'aggregator': 'min',
    'batch_size': 64,#0,
    'beta': 0.4,
    # 'beta_scheduler': 'none',
    'buffer_size': 100_000,
    # 'final_beta_multiplier': 6,
    'beta_end': 0.4,
    'gradient_steps': 1,
    'hidden_dim': 256,
    'learning_rate': 1e-3,
    'learning_starts': 15_000,
    'target_update_interval': 325,
    'tau': 0.3,
    'tau_theta': 0.96,
    'theta_update_interval': 500,
    'train_freq': 1,
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
    'learning_starts': 1000,
    'train_freq': 16,

}

# Set up a table of algos/envs to configs:
cartpoles = {
    'logu': cartpole_hparams2,
    'dqn': cartpole_dqn,
    'ppo': cartpole_ppo
}

acrobots = {
    'logu': acrobot_logu,
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
    'ppo': lunar_ppo,
    'dqn': lunar_dqn
}
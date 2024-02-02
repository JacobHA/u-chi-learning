
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

nature_pong = {
  "batch_size": 64,
  "beta": 0.87,
  "buffer_size": 1_000_000,
  "tau": 1,
  "train_freq": 4,
  "learning_starts": 50000 ,
  "learning_rate": 0.0001,#00025 ,
#   "gradient_momentum": 0.95 ,
#   "squared_gradient_momentum": 0.95 ,
#   "min_squared_gradient": 0.01 ,
#   "action_history_len": 4 ,
#   "action_repeat": 4 ,
#   "discount_factor": 0.99 ,
  "target_update_interval": 15000,
  "tau_theta": 0.98,
  'aggregator': 'max',
  'hidden_dim': 512,
}

cartpole_u = {
    'batch_size': 64,
    'beta': 0.7,
    'buffer_size': 50_000,
    'hidden_dim': 128,
    'learning_rate': 0.0035,
    'learning_starts': 0,#0.0025*50_000,
    'target_update_interval': 50,
    'tau': 0.73,
    'tau_theta': 0.76,
    'theta_update_interval': 10,#750,
    'train_freq': 1,
    'gradient_steps': 1,
    'aggregator': 'max',
    'prior_tau': 0.88,
    'prior_update_interval': 5000,
}

maze = {
    'batch_size': 256,
    'beta': 0.5,
    'buffer_size': 100_000,
    'hidden_dim': 64,
    'learning_rate': 0.005,
    'learning_starts': 0.4*50_000,
    'target_update_interval': 500,
    'tau': 0.9,
    'tau_theta': 0.9,
    'theta_update_interval': 200,#750,
    'train_freq': 1,
    'aggregator': 'max'
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
    'aggregator': 'max',
    'prior_update_interval': 500,
    'prior_tau': 0.42
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
    'target_update_interval': 600,
    'learning_starts': 1000,
    'train_freq': 16,

}

mcar_u = {
    'batch_size': 128,
    'beta': 0.83,
    'buffer_size': 100_000,
    'gradient_steps': 16,
    'learning_rate': 0.001,
    'learning_starts': 0.04*100_000,
    'target_update_interval': 50,
    'tau': 0.72,
    'tau_theta': 0.87,
    'theta_update_interval': 10,#750,
    'train_freq': 16,
    'hidden_dim': 32,
    'aggregator': 'max',
    'prior_tau': 0.62,
    'prior_update_interval': 5000,
}

mcar_sql = {
    'batch_size': 128,
    'beta': 0.7,
    'gamma': 0.99,
    'hidden_dim': 64,
    'learning_rate': 0.002,
    'learning_starts': 0.09*100_000,
    'target_update_interval': 100,
    'tau': 0.97,
    'gradient_steps': 2,
    'train_freq': 2,
}

pong_dqn = {
    'batch_size': 32,
    'buffer_size': 1_000_000,
    'exploration_final_eps': 0.01,
    'exploration_fraction': 0.1,
    'gamma': 0.99,
    'gradient_steps': 4,
    'hidden_dim': 256,
    'learning_rate': 0.00025,
    'learning_starts': 1000,
    'target_update_interval': 1000,
    'train_freq': 4,

}

sql_acro = {
    'batch_size': 128,
    'beta': 2.6,
    'gamma': 0.999,
    'hidden_dim': 32,
    'learning_rate': 0.0066,
    'learning_starts': 0.04*50_000,
    'target_update_interval': 100,
    'tau': 0.92,
    'train_freq': 9,
    'gradient_steps': 9,
}

sql_cpole = {
    'batch_size': 64,
    'beta': 0.1,
    'gamma': 0.98,
    'hidden_dim': 64,
    'learning_rate': 0.02,
    'learning_starts': 0.02*50_000,
    'target_update_interval': 100,
    'tau': 0.95,
    'train_freq': 9,
    'gradient_steps': 9,
}


# Set up a table of algos/envs to configs:
cartpoles = {
    'u': cartpole_u,
    'dqn': cartpole_dqn,
    'sql': sql_cpole,
}

acrobots = {
    'u': acrobot_u,
    'dqn': acrobot_dqn,
    'sql': sql_acro,
}

mcars = {
    'u': mcar_u,
    'dqn': mcar_dqn,
    'sql': mcar_sql,
}

pongs = {
    'u': nature_pong,
    'dqn': pong_dqn
}

id_to_hparam_dicts = {
    'CartPole-v1': cartpoles,
    'Acrobot-v1': acrobots,
    'MountainCar-v0': mcars,
    'PongNoFrameskip-v4': pongs
}
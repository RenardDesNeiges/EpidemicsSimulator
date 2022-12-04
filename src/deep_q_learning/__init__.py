
from datetime import datetime
import torch.nn as nn

PARAMS = {
    'COUNTRY_WIDE_NAIVE': {
        'log': True,
        'mode': 'binary',
        'model': 'naive',
        'run_name': 'naive_agent' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config': 'config/switzerland.yaml',
        'reward_sample_rate': 1,
        'threshold':20000,
        'confine_time':4,
    },
    'COUNTRY_WIDE_DEBUG': {
        'log': True,
        'mode': 'binary',
        'run_name': 'country_wide_debug_agent' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config': 'config/switzerland.yaml',
        'model': 'DQN',
        'target_update_rate': 5,
        'reward_sample_rate': 1,
        'eval_rate': 20,
        'eval_samples': 2,
        'num_episodes': 1,
        'criterion':  nn.HuberLoss(),
        'lr':  5e-3,
        'epsilon': 0.7,
        'epsilon_decrease': 300,
        'epsilon_floor': 0.2,
        'gamma': 0.7,
        'buffer_size': 10000,
        'batch_size': 512,
    },
    'COUNTRY_WIDE_BINARY': {
        'log': True,
        'mode': 'binary',
        'run_name': 'country_wide_binary_agent' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config': 'config/switzerland.yaml',
        'model': 'DQN',
        'target_update_rate': 5,
        'reward_sample_rate': 1,
        'eval_rate': 20,
        'eval_samples': 10,
        'num_episodes': 300,
        'criterion':  nn.HuberLoss(),
        'lr':  5e-3,
        'epsilon': 0.7,
        'epsilon_decrease': 300,
        'epsilon_floor': 0.2,
        'gamma': 0.7,
        'buffer_size': 10000,
        'batch_size': 512,
    },
    'COUNTRY_WIDE_BINARY_TOGGLE': {
        'log': True,
        'mode': 'toggle',
        'run_name': 'country_wide_binary_toggle_agent' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config': 'config/switzerland.yaml',
        'model': 'DQN',
        'target_update_rate': 5,
        'reward_sample_rate': 1,
        'eval_rate': 20,
        'eval_samples': 10,
        'num_episodes': 300,
        'criterion':  nn.HuberLoss(),
        'lr':  5e-3,
        'epsilon': 0.7,
        'epsilon_decrease': 300,
        'epsilon_floor': 0.2,
        'gamma': 0.7,
        'buffer_size': 10000,
        'batch_size': 512
    },
    'COUNTRY_WIDE_MULTI_TOGGLE': {
        'log': True,
        'mode': 'multi',
        'run_name': 'country_wide_multiaction_agent' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config': 'config/switzerland.yaml',
        'model': 'DQN',
        'target_update_rate': 5,
        'reward_sample_rate': 1,
        'eval_rate': 20,
        'eval_samples': 10,
        'num_episodes': 300,
        'criterion':  nn.HuberLoss(),
        'lr':  5e-3,
        'epsilon': 0.7,
        'epsilon_decrease': 300,
        'epsilon_floor': 0.2,
        'gamma': 0.9,
        'buffer_size': 10000,
        'batch_size': 512
    },
    'DISTRIBUTED_DEBUG': {
        'log': True,
        'mode': 'binary',
        'run_name': 'decentralized_debug' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config': 'config/switzerland.yaml',
        'model': 'DQN',
        'target_update_rate': 5,
        'reward_sample_rate': 1,
        'eval_rate': 20,
        'eval_samples': 2,
        'num_episodes': 300,
        'criterion':  nn.HuberLoss(),
        'lr':  5e-3,
        'epsilon': 0.7,
        'epsilon_decrease': 300,
        'epsilon_floor': 0.2,
        'gamma': 0.7,
        'buffer_size': 10000,
        'batch_size': 512,
    },
    'DISTRIBUTED_BINARY': {
        'log': True,
        'mode': 'binary',
        'run_name': 'decentralized_binary_agents' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config': 'config/switzerland.yaml',
        'model': 'DQN',
        'target_update_rate': 5,
        'reward_sample_rate': 1,
        'eval_rate': 20,
        'eval_samples': 10,
        'num_episodes': 300,
        'criterion':  nn.HuberLoss(),
        'lr':  5e-3,
        'epsilon': 0.7,
        'epsilon_decrease': 300,
        'epsilon_floor': 0.2,
        'gamma': 0.7,
        'buffer_size': 10000,
        'batch_size': 512,
    },
    'DISTRIBUTED_BINARY_TOGGLE': {
        'log': True,
        'mode': 'toggle',
        'run_name': 'decentralized_binary_toggled_agents' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config': 'config/switzerland.yaml',
        'model': 'DQN',
        'target_update_rate': 5,
        'reward_sample_rate': 1,
        'eval_rate': 20,
        'eval_samples': 10,
        'num_episodes': 300,
        'criterion':  nn.HuberLoss(),
        'lr':  5e-3,
        'epsilon': 0.7,
        'epsilon_decrease': 300,
        'epsilon_floor': 0.2,
        'gamma': 0.7,
        'buffer_size': 10000,
        'batch_size': 512,
    },
    'DISTRIBUTED_MULTI_TOGGLE': {
        'log': True,
        'mode': 'multi',
        'run_name': 'decentralized_multiaction_agent' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config': 'config/switzerland.yaml',
        'model': 'DQN',
        'target_update_rate': 5,
        'reward_sample_rate': 1,
        'eval_rate': 20,
        'eval_samples': 10,
        'num_episodes': 300,
        'criterion':  nn.HuberLoss(),
        'lr':  5e-3,
        'epsilon': 0.7,
        'epsilon_decrease': 300,
        'epsilon_floor': 0.2,
        'gamma': 0.9,
        'buffer_size': 10000,
        'batch_size': 512
    },
}


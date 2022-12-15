"""Pytorch Deep Q Learning implementation. Written to run experiments on an epidemic containment environment (simulating covid policy).


"""

import torch.nn as nn
from datetime import datetime
from typing import Dict, Any
from deep_q_learning.trainer import AbstractTrainer
from deep_q_learning.preprocessors import *
from gym import spaces

PARAMS = {
    'NAIVE': {
        'log': True,
        'action_space_generator': get_binary_action_space,
        'observation_space_generator': get_observation_space,
        'action_preprocessor': binary_action_preprocessor,
        'observation_preprocessor': naive_observation_preprocessor,
        'model': 'naive',
        'agent': 'naive',
        'run_name': 'naive_agent' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config': 'config/switzerland.yaml',
        'reward_sample_rate': 1,
        'threshold':20000,
        'confine_time':4,
    },
    'DEBUG': {
        'log': True,
        'action_space_generator': get_binary_action_space,
        'observation_space_generator': get_observation_space,
        'action_preprocessor': binary_action_preprocessor,
        'observation_preprocessor': naive_observation_preprocessor,
        'run_name': 'country_wide_debug_agent' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config': 'config/switzerland.yaml',
        'model': 'DQN', # TODO : replace the string-passing by directly loading the class here
        'agent': 'DQL',
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
    'BINARY': {
        'log': True,
        'action_space_generator': get_binary_action_space,
        'observation_space_generator': get_observation_space,
        'action_preprocessor': binary_action_preprocessor,
        'observation_preprocessor': naive_observation_preprocessor,
        'run_name': 'country_wide_binary_agent' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config': 'config/switzerland.yaml',
        'model': 'DQN',
        'agent': 'DQL',
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
        'batch_size': 1024,
    },
    'BINARY_TOGGLE': {
        'log': True,
        'action_space_generator': get_binary_action_space,
        'observation_space_generator': get_toggle_observation_space,
        'action_preprocessor': binary_toggle_action_preprocessor,
        'observation_preprocessor': binary_toggle_observation_preprocessor,
        'run_name': 'country_wide_binary_toggle_agent' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config': 'config/switzerland.yaml',
        'model': 'DQN',
        'agent': 'DQL',
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
        'batch_size': 1024,
    },
    'MULTI_TOGGLE': {
        'log': True,
        'action_space_generator': get_multi_action_space,
        'observation_space_generator': get_toggle_observation_space,
        'action_preprocessor': multi_toggle_action_preprocessor,
        'observation_preprocessor': multi_toggle_observation_preprocessor,
        'run_name': 'country_wide_multiaction_toggle_agent' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config': 'config/switzerland.yaml',
        'model': 'DQN',
        'agent': 'DQL',
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
        'batch_size': 1024,
    },
    'MULTI_FACTORIZED': {
        ## Logging parameters
        'log': True,
        'run_name': 'country_wide_multiaction_factorized_agent' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        ## Environment parameters
        'action_space_generator': get_multi_binary_action_space,
        'observation_space_generator': get_observation_space,
        'action_preprocessor': multi_factor_action_preprocessor,
        'observation_preprocessor': naive_observation_preprocessor,
        'env_config': 'config/switzerland.yaml',
        ## Training parameters
        'model': 'FactoredDQN',
        'agent': 'FactoredDQL',
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
        'batch_size': 1024,
    },
}



def getTrainer()->AbstractTrainer:
    """Loads a trainer object for implementing learning on the environment.

    Args:
        name (str): name of the trainer object (valid values are "Trainer" and "DistributedTrainer").

    Raises:
        ValueError: raised when name is invalid

    Returns:
        Trainer: the trainer object
    """
    from deep_q_learning.trainer import Trainer
    return Trainer


def getParams(name:str)->Dict[str,Any]:
    """Loads default training parameters.

    Args:
        name (str): name of the parameter dict

    Raises:
        ValueError: raised whne name is invalid

    Returns:
        Dict(srt:any): the parameters dictionary
    """
    if name not in PARAMS.keys():
        raise ValueError('Invalid Parameters')
    else:
        return PARAMS[name]
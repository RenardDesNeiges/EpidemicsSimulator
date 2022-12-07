"""Action and observation preprocessor functions
"""

import torch
import numpy as np
from epidemic_env.dynamics import ModelDynamics
from epidemic_env.env import ACTION_CONFINE, ACTION_HOSPITAL, ACTION_ISOLATE, ACTION_VACCINATE, SCALE
from typing import Dict, Any
from gym import spaces

"""Action Preprocessors
"""

def binary_action_preprocessor(a:torch.Tensor, dyn:ModelDynamics):
    conf = (dyn.c_confined['Lausanne'] != 1)
    isol = (dyn.c_isolated['Lausanne'] != 1)
    vacc = (dyn.vaccinate['Lausanne'] != 0)
    hosp = (dyn.extra_hospital_beds['Lausanne'] != 1)
    conf = (a == 1)
    
    return {
        'confinement': conf,
        'isolation': isol,
        'hospital': hosp,
        'vaccinate': vacc,
    }

def binary_toggle_action_preprocessor(a:torch.Tensor, dyn:ModelDynamics):
    conf = (dyn.c_confined['Lausanne'] != 1)
    isol = (dyn.c_isolated['Lausanne'] != 1)
    vacc = (dyn.vaccinate['Lausanne'] != 0)
    hosp = (dyn.extra_hospital_beds['Lausanne'] != 1)
    
    conf = not conf
    
    return {
        'confinement': conf,
        'isolation': isol,
        'hospital': hosp,
        'vaccinate': vacc,
    }

def multi_toggle_action_preprocessor(a:torch.Tensor, dyn:ModelDynamics):
    conf = (dyn.c_confined['Lausanne'] != 1)
    isol = (dyn.c_isolated['Lausanne'] != 1)
    vacc = (dyn.vaccinate['Lausanne'] != 0)
    hosp = (dyn.extra_hospital_beds['Lausanne'] != 1)
    
    if a == ACTION_CONFINE:
        conf = not conf
    elif a == ACTION_ISOLATE:
        isol = not isol
    elif a == ACTION_VACCINATE:
        vacc = not vacc
    elif a == ACTION_HOSPITAL:
        hosp = not hosp
    
    return {
        'confinement': conf,
        'isolation': isol,
        'hospital': hosp,
        'vaccinate': vacc,
    }

def multi_factor_action_preprocessor(a:torch.Tensor, dyn:ModelDynamics):
    raise NotImplemented

"""Observation Preprocessors
"""

def binary_observation_preprocessor(obs_dict:Dict[str,Any], dyn:ModelDynamics):
    infected = SCALE * \
        np.array([np.array(obs_dict['city']['infected'][c]) /
                    obs_dict['pop'][c] for c in dyn.cities])
    dead = SCALE * \
        np.array([np.array(obs_dict['city']['dead'][c])/obs_dict['pop'][c]
                    for c in dyn.cities])
    return torch.Tensor(np.stack((infected, dead))).unsqueeze(0)

def binary_toggle_observation_preprocessor(obs_dict:Dict[str,Any], dyn:ModelDynamics):
    infected = SCALE * \
        np.array([np.array(obs_dict['city']['infected'][c]) /
                    obs_dict['pop'][c] for c in dyn.cities])
    dead = SCALE * \
        np.array([np.array(obs_dict['city']['dead'][c])/obs_dict['pop'][c]
                    for c in dyn.cities])
    confined = np.ones_like(
        dead)*int((dyn.c_confined['Lausanne'] != 1))
    return torch.Tensor(np.stack((infected, dead, confined))).unsqueeze(0)

def multi_toggle_observation_preprocessor(obs_dict:Dict[str,Any], dyn:ModelDynamics):
    infected = SCALE * \
        np.array([np.array(obs_dict['city']['infected'][c]) /
                    obs_dict['pop'][c] for c in dyn.cities])
    dead = SCALE * \
        np.array([np.array(obs_dict['city']['dead'][c])/obs_dict['pop'][c]
                    for c in dyn.cities])
    self_obs = np.concatenate((
        np.ones((1, 7)) * int((dyn.c_confined['Lausanne'] != 1)),
        np.ones((1, 7)) * int((dyn.c_isolated['Lausanne'] != 1)),
        np.ones((1, 7)) * int((dyn.vaccinate['Lausanne'] != 0)),
        np.ones((1, 7)) *
        int((dyn.extra_hospital_beds['Lausanne'] != 1)), np.zeros((5, 7))
    ))
    return torch.Tensor(np.stack((infected, dead, self_obs))).unsqueeze(0)

def multi_factor_observation_preprocessor(obs_dict:Dict[str,Any], dyn:ModelDynamics):
    raise NotImplemented

"""Action/Observation space generators
"""

def get_binary_action_space(dyn:ModelDynamics):
    return spaces.Discrete(2)

def get_multi_action_space(dyn:ModelDynamics):
    return spaces.Discrete(5)

def get_multi_binary_action_space(dyn:ModelDynamics):
    return spaces.MultiBinary(4)

def get_binary_observation_space(dyn:ModelDynamics):
    return spaces.Box(
                low=0,
                high=1,
                shape=(2, dyn.n_cities, dyn.env_step_length),
                dtype=np.float16)

def get_toggle_binary_observation_space(dyn:ModelDynamics):
    return spaces.Box(
                low=0,
                high=1,
                shape=(3, dyn.n_cities, dyn.env_step_length),
                dtype=np.float16)

def get_multi_toggle_observation_space(dyn:ModelDynamics):
    return spaces.Box(
                low=0,
                high=1,
                shape=(3, dyn.n_cities, dyn.env_step_length),
                dtype=np.float16)

def get_multi_factored_observation_space(dyn:ModelDynamics):
    raise NotImplemented
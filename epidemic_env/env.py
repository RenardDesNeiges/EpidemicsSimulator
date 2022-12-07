"""Custom Environment that subclasses gym env."""

import gym
import torch
from gym.spaces import Space
import torch
from epidemic_env.dynamics import ModelDynamics
from datetime import datetime as dt
from collections import namedtuple
from typing import Dict, Tuple, Any, Callable

ACTION_CONFINE = 1
ACTION_ISOLATE = 2
ACTION_HOSPITAL = 3
ACTION_VACCINATE = 4
SCALE = 100 #TODO : abstract that constant away

CONST_REWARD = 7
DEATH_COST = 7e4
ANN_COST = 6
ISOL_COST = 1.5
CONF_COST = 6
VACC_COST = 0.08
HOSP_COST = 1



RewardTuple = namedtuple('RewardTuple',['reward','dead','conf','ann','vacc','hosp','isol'])

class Env(gym.Env):
    """Environment class, subclass of [gym.Env](https://www.gymlibrary.dev)."""
    metadata = {'render.modes': ['human']}
    
    def __init__(self,  source_file:str,
                        action_space:Space, 
                        observation_space:Space,
                        ep_len:int=30, 
                        action_preprocessor:Callable=lambda x:x, 
                        observation_preprocessor:Callable=lambda x:x, 
                        )->None:
        """**TODO describe:**
        Action Spaces (per mode)

        Modes 'binary', 'toggle, 'multi', 'factored' ==> TODO : Remove cases, use preprocessing functions
        
        pass -> preprocessor function
        
        pass -> output action space

        Args:
            source_file (str): _description_
            ep_len (int, optional): _description_. Defaults to 30.
            action_preprocessor (_type_, optional): _description_. Defaults to lambdax:x.
            observation_preprocessor (_type_, optional): _description_. Defaults to lambdax:x.
            action_space (Space, optional): _description_. Defaults to None.
            mode (str, optional): _description_. Defaults to 'binary'.
        """
        
        super(Env, self).__init__()

        self.ep_len = ep_len
        self.dyn = ModelDynamics(source_file)  # create the dynamical model
    
        self.action_space = action_space
        self.observation_space = observation_space
            
        self.action_preprocessor        = action_preprocessor
        self.observation_preprocessor   = observation_preprocessor
        
        self.reward = torch.Tensor([0]).unsqueeze(0)
        self.reset()

    def compute_reward(self, obs_dict:Dict[str,Dict[str,float]])->RewardTuple:
        """Computes the reward from an observation dictionary:
        
        $$
            \\begin{aligned}
            R(s,a) =  R_\\text{const}
            - A_\\text{cost} - V_\\text{cost} - H_\\text{cost} \\\\
            -\sum_{i \in C} 
            \\frac{\\text{pop}_i}{\\text{pop}_\\text{total}} 
            (
                D_\\text{cost} \cdot \Delta d_i  +
                C_\\text{cost} \cdot c_i +
                I_\\text{cost} \cdot i_i
            ) 
            \\\\
            R_\\text{const}\\text{ : constant reward}\\\\
            D_\\text{cost} \\text{ : cost of deaths}\\\\
            C_\\text{cost} \\text{ : cost of confinement time}\\\\
            I_\\text{cost} \\text{ : cost of isolation time}\\\\
            A_\\text{cost} \\text{ : cost of announcing confinement or isolation} \\\\
            V_\\text{cost} \\text{ : cost of a vaccination campaign}\\\\
            H_\\text{cost} \\text{ : cost of adding exceptional hospital beds}\\\\
        \end{aligned}
        $$

        Args:
            obs_dict (Dict[str,Dict[str,float]]): The observation.

        Returns:
            RewardTuple: the reward and all of it's components
        """
        def compute_death_cost():
            dead = 0
            for city in self.dyn.cities:
                if len(obs_dict['city']['dead'][city]) > 1:
                    dead += DEATH_COST * \
                        (obs_dict['city']['dead'][city][-1] - obs_dict['city']
                         ['dead'][city][-2]) / (self.dyn.total_pop)
                else:
                    dead += DEATH_COST * \
                        obs_dict['city']['dead'][city][-1] / \
                        (self.dyn.total_pop)
            return dead
        def compute_isolation_cost():
            isol = 0
            for city in self.dyn.cities:
                isol += ISOL_COST * \
                    int(self.dyn.c_isolated[city] == self.dyn.isolation_effectiveness) * \
                    obs_dict['pop'][city] / (self.dyn.total_pop)
            return isol
        def compute_confinement_cost():
            conf = 0
            for city in self.dyn.cities:
                conf += CONF_COST * \
                    int(self.dyn.c_confined[city] == self.dyn.confinement_effectiveness) * \
                    obs_dict['pop'][city] / (self.dyn.total_pop)
            return conf
        def compute_annoucement_cost():
            announcement = 0
            if self._get_info()['action']['confinement'] and not self.last_info['action']['confinement']:
                announcement += ANN_COST
            if self._get_info()['action']['isolation'] and not self.last_info['action']['isolation']:
                announcement += ANN_COST
            if self._get_info()['action']['vaccinate'] and not self.last_info['action']['vaccinate']:
                announcement += ANN_COST
            return announcement
        def compute_vaccination_cost():
            vacc = int(self.dyn.vaccinate['Lausanne'] != 0) * VACC_COST
            return vacc
        def compute_hospital_cost():
            hosp = (self.dyn.extra_hospital_beds['Lausanne'] != 1) * HOSP_COST
            return hosp

        dead = compute_death_cost()
        conf = compute_confinement_cost()
        ann = compute_annoucement_cost()
        vacc = compute_vaccination_cost()
        hosp = compute_hospital_cost()
        isol = compute_isolation_cost()

        rew = CONST_REWARD - dead - conf - ann - vacc - hosp
        return RewardTuple(torch.Tensor([rew]).unsqueeze(0), dead, conf, ann, vacc, hosp, isol)

    def get_obs(self, obs_dict:Dict[str,Any])->torch.Tensor:
        """Generates an observation tensor from a dictionary of observations.

        Args:
            obs_dict (Dict[Any]): the observations dictionary.

        Raises:
            Exception: when the mode is incorrectly implemented.

        Returns:
            torch.Tensor: the observation tensor.
        """
        return self.observation_preprocessor(obs_dict,self.dyn)
    
    def _parse_action(self, a):        
        return self.action_preprocessor(a,self.dyn)
        
    def _get_info(self)->Dict[str,Any]:
        """Grabs the dynamical system information dictionary from the simulator.

        Returns:
            Dict[str,Any]: The information dictionary.
        """
        info = {
            'parameters': self.dyn.epidemic_parameters(self.day),
            'action': {
                'confinement': (self.dyn.c_confined['Lausanne'] != 1),
                'isolation': (self.dyn.c_isolated['Lausanne'] != 1),
                'vaccinate': (self.dyn.vaccinate['Lausanne'] != 0),
                'hospital': (self.dyn.extra_hospital_beds['Lausanne'] != 1),
            },
            'dead_cost': self.dead_cost,
            'conf_cost': self.conf_cost,
            'ann_cost': self.ann_cost,
            'vacc_cost': self.vacc_cost,
            'hosp_cost': self.hosp_cost,
        }
        return info

    def step(self, action:int)->Tuple[torch.Tensor,torch.Tensor,Dict[str,Any]]:
        """Perform one environment step.

        Args:
            action (int): the action

        Returns:
            Tuple[torch.Tensor,torch.Tensor,Dict[str,Any]]: A tuple containing
            - in element 1
        """
        self.day += 1
        self.last_action = action
        self.last_info = self._get_info()
        for c in self.dyn.cities:
            self.dyn.set_action(self._parse_action(action), c)
        _obs_dict = self.dyn.step()
        self.last_obs = self.get_obs(_obs_dict)

        r = self.compute_reward(_obs_dict)
        self.reward     = r.reward
        self.dead_cost  = r.dead
        self.conf_cost  = r.conf
        self.ann_cost   = r.ann
        self.vacc_cost  = r.vacc
        self.hosp_cost  = r.hosp
        self.isol       = r.isol

        done = self.day >= self.ep_len
        return self.last_obs, self.reward, done, self._get_info()

    def reset(self, seed:int=None)->Tuple[torch.Tensor,Dict[str,Any]]:
        """Reset the state of the environment to an initial state

        Args:
            seed (int, optional): random seed (for reproductability). Defaults to None.

        Returns:
            Tuple[torch.Tensor,Dict[str,Any]]: a tuple containing, in element 0 the observation tensor, in element 1 the information dictionary
        """
        self.last_action = 0
        self.day = 0
        self.dead_cost = 0
        self.conf_cost = 0
        self.ann_cost = 0
        self.vacc_cost = 0
        self.hosp_cost = 0
        self.isol = 0
        self.dyn.reset()
        if seed is None:
            self.dyn.start_epidemic(dt.now())
        else:
            self.dyn.start_epidemic(seed)

        _obs_dict = self.dyn.step()
        self.last_obs = self.get_obs(_obs_dict)
        self.last_info = self._get_info()
        return self.last_obs, self.last_info

    def render(self, mode='human', close=False):
        total, _ = self.dyn.epidemic_parameters(self.day)
        print('Epidemic state : \n   - dead: {}\n   - infected: {}'.format(
            total['dead'], total['infected']))



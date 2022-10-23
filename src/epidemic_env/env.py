import gym
import torch
from gym import spaces
import numpy as np
from epidemic_env.dynamics import ModelDynamics
from datetime import datetime as dt

ACTION_CONFINE = 1
ACTION_ISOLATE = 2
ACTION_HOSPITAL = 3
ACTION_VACCINATE = 4
SCALE = 100

"""Custom Environment that subclasses gym env"""
class CountryWideEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, source_file, ep_len=100, mode='binary'):
        super(CountryWideEnv, self).__init__()    
        
        self.ep_len = ep_len
        self.dyn = ModelDynamics(source_file) # create the dynamical model
        self.mode = mode
        
        if self.mode == 'toggle':
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(
                low=0, 
                high=1, 
                shape=(3, self.dyn.n_cities, self.dyn.env_step_length),
                dtype=np.float16)
        elif self.mode == 'binary':
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(
                low=0, 
                high=1, 
                shape=(2, self.dyn.n_cities, self.dyn.env_step_length), 
                dtype=np.float16)
        elif self.mode == 'multi':
            self.action_space = spaces.Discrete(5) # 4 actions + do nothing
            self.observation_space = spaces.Box(
                low=0, 
                high=1, 
                shape=(3, self.dyn.n_cities, self.dyn.env_step_length), 
                dtype=np.float16)
        else:
            raise Exception(NotImplemented)

        self.reward = torch.Tensor([0]).unsqueeze(0)
        self.reset()
    

    def compute_reward(self, obs_dict):
        
        def compute_death_cost():
            dead = 0
            for city in self.dyn.cities:
                if len(obs_dict['city']['dead'][city]) > 1:
                    dead +=  7e4 * (obs_dict['city']['dead'][city][-1] - obs_dict['city']['dead'][city][-2] ) / (self.dyn.total_pop)
                else:
                    dead +=  7e4 * obs_dict['city']['dead'][city][-1] / (self.dyn.total_pop)
            return dead
                
        def compute_isolation_cost():
            isol = 0
            for city in self.dyn.cities:
                isol +=  1.5 * int(self.dyn.c_isolated[city] == self.dyn.isolation_effectiveness)*obs_dict['pop'][city]  / (self.dyn.total_pop)
            return isol
                 
        def compute_confinement_cost():
            conf = 0
            for city in self.dyn.cities:
                conf +=  2 * int(self.dyn.c_confined[city] == self.dyn.confinement_effectiveness)*obs_dict['pop'][city]  / (self.dyn.total_pop)
            return conf
        
        def compute_annoucement_cost():
            announcement = 0 
            if not (self.dyn.c_confined[city] == self.dyn.confinement_effectiveness) and self.last_action == ACTION_CONFINE:
                announcement = 2
            if not (self.dyn.c_isolated[city] == self.dyn.isolation_effectiveness) and self.last_action == ACTION_ISOLATE: 
                announcement = 2
            return announcement 
        
        def compute_vaccination_cost():
            vacc = int(self.dyn.vaccinate['Lausanne'] != 0) * 0.08
            return vacc

        def compute_hospital_cost():
            hosp = (self.dyn.extra_hospital_beds['Lausanne'] != 1)*1
            return hosp
        
        dead = compute_death_cost()
        conf = compute_confinement_cost()
        if self.mode == 'toggle':
            ann = compute_annoucement_cost()
            rew = 3 - dead - conf - ann
            return torch.Tensor([rew]).unsqueeze(0), dead, conf, ann
        elif self.mode == 'binary':
            rew = 3 - dead - conf 
            return torch.Tensor([rew]).unsqueeze(0), dead, conf
        elif self.mode == 'multi':
            ann = compute_annoucement_cost()
            vacc = compute_vaccination_cost()
            isol = compute_isolation_cost()
            hosp = compute_hospital_cost()
            rew = 3 - dead - conf - ann - vacc - hosp
            return torch.Tensor([rew]).unsqueeze(0), dead, conf, ann, vacc, hosp, isol
        else:
            raise Exception(NotImplemented)

    def get_obs(self, obs):
        if self.mode == 'binary':
            infected = SCALE*np.array([np.array(obs['city']['infected'][c])/obs['pop'][c] for c in self.dyn.cities])
            dead = SCALE*np.array([np.array(obs['city']['dead'][c])/obs['pop'][c] for c in self.dyn.cities])
            return torch.Tensor(np.stack((infected,dead))).unsqueeze(0)
        elif self.mode == 'toggle':
            infected = SCALE*np.array([np.array(obs['city']['infected'][c])/obs['pop'][c] for c in self.dyn.cities])
            dead = SCALE*np.array([np.array(obs['city']['dead'][c])/obs['pop'][c] for c in self.dyn.cities])
            confined = np.ones_like(dead)*int((self.dyn.c_confined['Lausanne'] != 1))
            return torch.Tensor(np.stack((infected,dead,confined))).unsqueeze(0)
        elif self.mode == 'multi':
            infected = SCALE*np.array([np.array(obs['city']['infected'][c])/obs['pop'][c] for c in self.dyn.cities])
            dead = SCALE*np.array([np.array(obs['city']['dead'][c])/obs['pop'][c] for c in self.dyn.cities])
            self_obs =  np.concatenate((
                np.ones((1,7)) * int((self.dyn.c_confined['Lausanne'] != 1)),
                np.ones((1,7)) * int((self.dyn.c_isolated['Lausanne'] != 1)),
                np.ones((1,7)) * int((self.dyn.vaccinate['Lausanne'] != 0)),
                np.ones((1,7)) * int((self.dyn.extra_hospital_beds['Lausanne'] != 1)),
                np.zeros((5,7))
            ))
            return torch.Tensor(np.stack((infected,dead,self_obs))).unsqueeze(0)
        else:
            raise Exception(NotImplemented)

    def parse_action(self,a):
        if self.mode == 'binary':
            return {
                'confinement': a==1,
                'isolation': False,
                'hospital': False,
                'vaccinate': False,
            }
        elif self.mode == 'toggle':
            conf = (self.dyn.c_confined['Lausanne'] != 1)
            if a == ACTION_CONFINE :
                conf = not conf
            return {
                'confinement': conf,
                'isolation': False,
                'hospital': False,
                'vaccinate': False,
            }
        elif self.mode == 'multi':
            conf = (self.dyn.c_confined['Lausanne'] != 1)
            isol = (self.dyn.c_isolated['Lausanne'] != 1)
            vacc = (self.dyn.vaccinate['Lausanne'] != 0)
            hosp = (self.dyn.extra_hospital_beds['Lausanne'] != 1)
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
                'hospital': vacc,
                'vaccinate': hosp,
            }
        else:
            raise Exception(NotImplemented)

    def get_info(self):
        info = {
                'parameters':self.dyn.epidemic_parameters(self.day),
                'action': {
                    'confinement': (self.dyn.c_confined['Lausanne'] != 1),
                    'isolation': (self.dyn.c_isolated['Lausanne'] != 1),
                    'vaccinate': (self.dyn.vaccinate['Lausanne'] != 0),
                    'hospital': (self.dyn.extra_hospital_beds['Lausanne'] != 1),
                    },
                'dead_cost': self.dead_cost,
                'conf_cost': self.conf_cost,
                }
        if self.mode == 'binary':
            return info
        elif self.mode == 'toggle':
            info['ann_cost'] = self.ann_cost
            return info
        elif self.mode == 'multi':
            info['ann_cost'] = self.ann_cost # TODO deal with secondary action costs
            info['vacc_cost'] = self.vacc_cost # TODO deal with secondary action costs
            info['hosp_cost'] = self.hosp_cost # TODO deal with secondary action costs
            return info
        else:
            raise Exception(NotImplemented)

    # Execute one time step within the environment
    def step(self, action):
        self.day += 1
        self.last_action = action
        for c in self.dyn.cities:
            self.dyn.set_action(self.parse_action(action),c)
        _obs_dict = self.dyn.step()
        self.last_obs = self.get_obs(_obs_dict)

        if self.mode == 'binary':
            self.reward, self.dead_cost, self.conf_cost = self.compute_reward(_obs_dict)
        elif self.mode=='toggle':
            self.reward, self.dead_cost, self.conf_cost, self.ann_cost = self.compute_reward(_obs_dict)
        elif self.mode=='multi':
            self.reward, self.dead_cost, self.conf_cost, self.ann_cost, self.vacc_cost, self.hosp_cost, self.isol = self.compute_reward(_obs_dict)
        else:
            raise Exception(NotImplemented)
        
        self.total_reward += self.reward
        done = self.day >= self.ep_len
        return self.last_obs, self.reward, done, self.get_info()

    # Reset the state of the environment to an initial state
    def reset(self, seed = None):
        self.last_action = 0
        self.day = 0
        self.total_reward = 0
        self.dead_cost = 0
        self.conf_cost = 0
        if self.mode == 'toggle':
            self.ann_cost = 0
        if self.mode == 'multi':
            self.ann_cost = 0
            self.vacc_cost = 0
            self.hosp_cost = 0
            self.isol = 0   
        self.dyn.reset()
        if seed is None:
            self.dyn.start_epidemic(dt.now())
        else:
            self.dyn.start_epidemic(seed)
            
        _obs_dict = self.dyn.step() # Eyo c'est un tuple ça
        self.last_obs = self.get_obs(_obs_dict)
        return self.last_obs, self.get_info()

    # Render the environment to the screen 
    def render(self, mode='human', close=False):
        total, _ = self.dyn.epidemic_parameters(self.day)
        print('Epidemic state : \n   - dead: {}\n   - infected: {}'.format(total['dead'],total['infected']))
        
        
"""Custom Environment that subclasses gym env (for distributed multi agent learning)"""
class DistributedEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, source_file, ep_len=100, mode='binary'):
        super(DistributedEnv, self).__init__()    
        
        self.ep_len = ep_len
        self.dyn = ModelDynamics(source_file) # create the dynamical model
        self.mode = mode
        
        if self.mode == 'toggle':
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(
                low=0, 
                high=1, 
                shape=(3, self.dyn.n_cities, self.dyn.env_step_length),
                dtype=np.float16)
        elif self.mode == 'binary':
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(
                low=0, 
                high=1, 
                shape=(2, self.dyn.n_cities, self.dyn.env_step_length), 
                dtype=np.float16)
        elif self.mode == 'multi':
            self.action_space = spaces.Discrete(5) # 4 actions + do nothing
            self.observation_space = spaces.Box(
                low=0, 
                high=1, 
                shape=(3, self.dyn.n_cities, self.dyn.env_step_length), 
                dtype=np.float16)
        else:
            raise Exception(NotImplemented)

        self.rewards = {c:torch.Tensor([0]).unsqueeze(0) for c in self.dyn.cities}
        self.reset()
    

    def compute_reward(self, city, obs_dict):
        
        def compute_death_cost():
            if len(obs_dict['city']['dead'][city]) > 1:
                dead =  7e4 * (obs_dict['city']['dead'][city][-1] - obs_dict['city']['dead'][city][-2] ) / (self.dyn.total_pop)
            else:
                dead =  7e4 * obs_dict['city']['dead'][city][-1] / (self.dyn.total_pop)
            return dead
                
        def compute_isolation_cost():
            isol =  1.5 * int(self.dyn.c_isolated[city] == self.dyn.isolation_effectiveness)*obs_dict['pop'][city]  / (self.dyn.total_pop)
            return isol
                 
        def compute_confinement_cost():
            conf =  2 * int(self.dyn.c_confined[city] == self.dyn.confinement_effectiveness)*obs_dict['pop'][city]  / (self.dyn.total_pop)
            return conf
        
        def compute_annoucement_cost():
            if not (self.dyn.c_confined[city] == self.dyn.confinement_effectiveness) and self.last_action == ACTION_CONFINE:
                announcement = 2
            if not (self.dyn.c_isolated[city] == self.dyn.isolation_effectiveness) and self.last_action == ACTION_ISOLATE: 
                announcement = 2
            return announcement 
        
        def compute_vaccination_cost():
            vacc = int(self.dyn.vaccinate['Lausanne'] != 0) * 0.08
            return vacc

        def compute_hospital_cost():
            hosp = (self.dyn.extra_hospital_beds['Lausanne'] != 1)*1
            return hosp
        
        dead = compute_death_cost()
        conf = compute_confinement_cost()
        if self.mode == 'toggle':
            ann = compute_annoucement_cost()
            rew = 3 - dead - conf - ann
            return torch.Tensor([rew]).unsqueeze(0), dead, conf, ann
        elif self.mode == 'binary':
            rew = 3 - dead - conf 
            return torch.Tensor([rew]).unsqueeze(0), dead, conf
        elif self.mode == 'multi':
            ann = compute_annoucement_cost()
            vacc = compute_vaccination_cost()
            isol = compute_isolation_cost()
            hosp = compute_hospital_cost()
            rew = 3 - dead - conf - ann - vacc - hosp
            return torch.Tensor([rew]).unsqueeze(0), dead, conf, ann, vacc, hosp, isol
        else:
            raise Exception(NotImplemented)

    def get_obs(self, obs):
        if self.mode == 'binary':
            infected = SCALE*np.array([np.array(obs['city']['infected'][c])/obs['pop'][c] for c in self.dyn.cities])
            dead = SCALE*np.array([np.array(obs['city']['dead'][c])/obs['pop'][c] for c in self.dyn.cities])
            return torch.Tensor(np.stack((infected,dead))).unsqueeze(0)
        elif self.mode == 'toggle':
            infected = SCALE*np.array([np.array(obs['city']['infected'][c])/obs['pop'][c] for c in self.dyn.cities])
            dead = SCALE*np.array([np.array(obs['city']['dead'][c])/obs['pop'][c] for c in self.dyn.cities])
            confined = np.ones_like(dead)*int((self.dyn.c_confined['Lausanne'] != 1))
            return torch.Tensor(np.stack((infected,dead,confined))).unsqueeze(0)
        elif self.mode == 'multi':
            infected = SCALE*np.array([np.array(obs['city']['infected'][c])/obs['pop'][c] for c in self.dyn.cities])
            dead = SCALE*np.array([np.array(obs['city']['dead'][c])/obs['pop'][c] for c in self.dyn.cities])
            self_obs =  np.concatenate((
                np.ones((1,7)) * int((self.dyn.c_confined['Lausanne'] != 1)),
                np.ones((1,7)) * int((self.dyn.c_isolated['Lausanne'] != 1)),
                np.ones((1,7)) * int((self.dyn.vaccinate['Lausanne'] != 0)),
                np.ones((1,7)) * int((self.dyn.extra_hospital_beds['Lausanne'] != 1)),
                np.zeros((5,7))
            ))
            return torch.Tensor(np.stack((infected,dead,self_obs))).unsqueeze(0)
        else:
            raise Exception(NotImplemented)

    def parse_action(self,a):
        if self.mode == 'binary':
            return {
                'confinement': a==1,
                'isolation': False,
                'hospital': False,
                'vaccinate': False,
            }
        elif self.mode == 'toggle':
            conf = (self.dyn.c_confined['Lausanne'] != 1)
            if a == ACTION_CONFINE :
                conf = not conf
            return {
                'confinement': conf,
                'isolation': False,
                'hospital': False,
                'vaccinate': False,
            }
        elif self.mode == 'multi':
            conf = (self.dyn.c_confined['Lausanne'] != 1)
            isol = (self.dyn.c_isolated['Lausanne'] != 1)
            vacc = (self.dyn.vaccinate['Lausanne'] != 0)
            hosp = (self.dyn.extra_hospital_beds['Lausanne'] != 1)
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
                'hospital': vacc,
                'vaccinate': hosp,
            }
        else:
            raise Exception(NotImplemented)

    def get_info(self):
        info = {
                'parameters':self.dyn.epidemic_parameters(self.day),
                'action': {
                    'confinement': {c:(self.dyn.c_confined[c] != 1) for c in self.dyn.cities},
                    'isolation': {c:(self.dyn.c_isolated['Lausanne'] != 1) for c in self.dyn.cities},
                    'vaccinate': {c:(self.dyn.vaccinate['Lausanne'] != 0) for c in self.dyn.cities},
                    'hospital': {c:(self.dyn.extra_hospital_beds['Lausanne'] != 1) for c in self.dyn.cities},
                    },
                'dead_cost': self.dead_cost,
                'conf_cost': self.conf_cost,
                }
        if self.mode == 'binary':
            return info
        elif self.mode == 'toggle':
            info['ann_cost'] = self.ann_cost
            return info
        elif self.mode == 'multi':
            info['ann_cost'] = self.ann_cost # TODO deal with secondary action costs
            info['vacc_cost'] = self.vacc_cost # TODO deal with secondary action costs
            info['hosp_cost'] = self.hosp_cost # TODO deal with secondary action costs
            return info
        else:
            raise Exception(NotImplemented)

    # Execute one time step within the environment
    def step(self, actions):

        self.day += 1
        self.last_actions = actions
        for c in self.dyn.cities:
            self.dyn.set_action(self.parse_action(actions[c][0]),c)
        _obs_dict = self.dyn.step()
        self.last_obs = self.get_obs(_obs_dict)

        self.rewards = {}
        if self.mode == 'binary':
            self.dead_cost = 0
            self.conf_cost = 0
            for c in self.dyn.cities:
                reward, dead_cost, conf_cost = self.compute_reward(c,_obs_dict)
                self.rewards[c] = reward
                self.dead_cost += dead_cost
                self.conf_cost += conf_cost
        elif self.mode=='toggle':
            raise Exception(NotImplemented)
            self.rewards, self.dead_cost, self.conf_cost, self.ann_cost = self.compute_reward(c,_obs_dict)
        elif self.mode=='multi':
            raise Exception(NotImplemented)
            self.rewards, self.dead_cost, self.conf_cost, self.ann_cost, self.vacc_cost, self.hosp_cost, self.isol = self.compute_reward(c,_obs_dict)
        else:
            raise Exception(NotImplemented)
        
        self.total_reward = np.sum([r.detach().numpy() for (_,r) in self.rewards.items()])
        done = self.day >= self.ep_len
        return self.last_obs, self.rewards, done, self.get_info()

    # Reset the state of the environment to an initial state
    def reset(self, seed = None):
        self.last_action = 0
        self.day = 0
        self.total_reward = 0
        self.dead_cost = 0
        self.conf_cost = 0
        if self.mode == 'toggle':
            self.ann_cost = 0
        if self.mode == 'multi':
            self.ann_cost = 0
            self.vacc_cost = 0
            self.hosp_cost = 0
            self.isol = 0   
        self.dyn.reset()
        if seed is None:
            self.dyn.start_epidemic(dt.now())
        else:
            self.dyn.start_epidemic(seed)
            
        _obs_dict = self.dyn.step() # Eyo c'est un tuple ça
        self.last_obs = self.get_obs(_obs_dict)
        return self.last_obs, self.get_info()

    # Render the environment to the screen 
    def render(self, mode='human', close=False):
        total, _ = self.dyn.epidemic_parameters(self.day)
        print('Epidemic state : \n   - dead: {}\n   - infected: {}'.format(total['dead'],total['infected']))
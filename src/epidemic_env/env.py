from unittest import case
import gym
from soupsieve import match
import torch
from gym import spaces
import numpy as np
from epidemic_env.dynamics import ModelDynamics
from datetime import datetime as dt

SCALE = 100
"""Custom Environment that subclasses gym env"""
class EpidemicEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, source_file, ep_len=100):
        super(EpidemicEnv, self).__init__()    
        
        self.ep_len = ep_len
        
        self.dyn = ModelDynamics(source_file) # create the dynamical model
        
        # action space (any combination of 4 actions)
        N_DISCRETE_ACTIONS = 5 #4#2**4
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS) # toggle behavior


        # the observation space is of shape N_CHANNEL x N_DAYS
        #                                 = 2 x 7
        # note that values are normalized between 0 and 1
        self.observation_space = spaces.Box(
                        low=0, 
                        high=1, 
                        shape=(2, self.dyn.env_step_length), 
                        dtype=np.float16)
        
        self.reward = torch.Tensor([0]).unsqueeze(0)
        self.reset()
    
    # Compute a reward
    def compute_reward(self, obs_dict):
        # TODO : per city reward
        dead_penality = 1e5 * obs_dict['total']['dead'][-1]/self.dyn.total_pop
        confinement_penality = 150*np.dot(
            np.array([int(e) for (_,e) in self.dyn.c_confined]), 
            np.array([float(e) for (_,e) in obs_dict['pop'].items()]))/self.dyn.total_pop
        isolation_penality = 50*np.dot(
            np.array([int(e) for (_,e) in self.dyn.c_isolated]), 
            np.array([float(e) for (_,e) in obs_dict['pop'].items()]))/self.dyn.total_pop
        hospital_penality = 200*np.dot(
            np.array([int(e) for (_,e) in self.dyn.extra_hospital_beds]), 
            np.array([float(e) for (_,e) in obs_dict['pop'].items()]))/self.dyn.total_pop
        vaccination_penality = 40*np.dot(
            np.array([int(e) for (_,e) in self.dyn.vaccinate]), 
            np.array([float(e) for (_,e) in obs_dict['pop'].items()]))/self.dyn.total_pop
        
        rew = (2000 - dead_penality - confinement_penality - isolation_penality - hospital_penality - vaccination_penality) / 1e5
        

        return torch.Tensor([rew]).unsqueeze(0)
    

    def compute_city_reward(self, city, obs_dict):
        # TODO : per city reward
        dead =  1e5 * obs_dict['total']['dead'][-1]
        conf =  150 * int(self.dyn.c_confined[city])*obs_dict['pop'][city]
        isol =  50  * int(self.dyn.c_isolated[city])*obs_dict['pop'][city]
        hosp =  200 * int(self.dyn.extra_hospital_beds[city]) * obs_dict['pop'][city]
        vacc =  40  * int(self.dyn.vaccinate[city]) * obs_dict['pop'][city]

        rew = (500 - dead - conf - isol - hosp - vacc) / (1e5 * self.dyn.total_pop)
        return torch.Tensor([rew]).unsqueeze(0)

    # TODO : update for the new obs
    # converts an action to an action dictionary
    # def vec2dict(self, act):
    #     act_digits = '{0:04b}'.format(act)
    #     act_dict = self.dyn.NULL_ACTION
    #     act_dict['confinement'] = {e:(act_digits[0] == '1') for (e,_) in act_dict['confinement'].items()}
    #     act_dict['isolation'] = {e:(act_digits[1] == '1') for (e,_) in act_dict['isolation'].items()}
    #     act_dict['hospital'] = {e:(act_digits[2] == '1') for (e,_) in act_dict['hospital'].items()}
    #     act_dict['vaccinate'] = (act_digits[3] == '1')
    #     return act_dict
    
    # TODO : update for the new obs    
    # converts a dictionary of observations to a normalized observation vector
    def dict2vec(self, obs):
        infected = SCALE*np.array([np.array(obs['city']['infected'][c])/obs['pop'][c] for c in self.dyn.cities])
        dead = SCALE*np.array([np.array(obs['city']['dead'][c])/obs['pop'][c] for c in self.dyn.cities])
        return torch.Tensor(np.stack((infected,dead))).unsqueeze(0)

    def parseaction(self,a):
        key = [None,'confinement','isolation','hospital','vaccinate']
        return key[a]


    # Execute one time step within the environment
    def step(self, action):
        
        self.day += 1
        for _id, c in enumerate(self.dyn.cities):
            self.dyn.toggle(self.parseaction(action[_id]),c)
        _obs_dict = self.dyn.step()
        
        obs = self.dict2vec(_obs_dict)
        self.last_obs = self.dict2vec(obs)
        self.reward = []
        for c in self.dyn.cities:
            self.reward.append(self.compute_city_reward(c,))

        self.total_reward += self.reward
        done = self.day >= self.ep_len
        
        return obs, self.reward, done, {'parameters':self.dyn.epidemic_parameters(self.day)}

    # Reset the state of the environment to an initial state
    def reset(self, seed = None):
        self.day = 0
        self.total_reward = 0
        self.dyn.reset()
        if seed is None:
            self.dyn.start_epidemic(dt.now())
        else:
            self.dyn.start_epidemic(seed)
            
        _obs = self.dyn.step(self.dyn.NULL_ACTION)
        self.last_obs = self.dict2vec(_obs)
        return self.observe()
    
    def observe(self,):
        return self.last_obs, self.reward, (self.day >= self.ep_len), {'parameters':self.dyn.epidemic_parameters(self.day)}

    # Render the environment to the screen 
    def render(self, mode='human', close=False):
        total, _ = self.dyn.epidemic_parameters(self.day)
        print('Epidemic state : \n   - dead: {}\n   - infected: {}'.format(total['dead'],total['infected']))
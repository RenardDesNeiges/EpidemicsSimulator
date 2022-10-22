import gym
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
                        shape=(3, self.dyn.n_cities, self.dyn.env_step_length), 
                        dtype=np.float16)
        
        self.reward = torch.Tensor([0]).unsqueeze(0)
        self.reset()
    

    def compute_city_reward(self, city, obs_dict):
        dead =  1e5 * obs_dict['total']['dead'][-1]
        conf =  150 * int(self.dyn.c_confined[city])*obs_dict['pop'][city]
        isol =  50  * int(self.dyn.c_isolated[city])*obs_dict['pop'][city]
        hosp =  200 * int(self.dyn.extra_hospital_beds[city]) * obs_dict['pop'][city]
        vacc =  40  * int(self.dyn.vaccinate[city]) * obs_dict['pop'][city]

        rew = (500 - dead - conf - isol - hosp - vacc) / (1e5 * self.dyn.total_pop)
        return torch.Tensor([rew]).unsqueeze(0)

    # converts a dictionary of observations to a normalized observation vector
    def get_obs(self, obs):
        obs_list = []
        for city in self.dyn.cities:
            infected = SCALE*np.array([np.array(obs['city']['infected'][c])/obs['pop'][c] for c in self.dyn.cities])
            dead = SCALE*np.array([np.array(obs['city']['dead'][c])/obs['pop'][c] for c in self.dyn.cities]) # SHAPED CITIES x DAYS = 9 x 7
            state = 2* np.array( # SHAPED 4xDAYS = 4
                    [   int(self.dyn.c_confined[city]),
                        int(self.dyn.c_isolated[city]),
                        int(self.dyn.extra_hospital_beds[city]),
                        int(self.dyn.vaccinate[city]),
                        0,0,0,0,0, # ugly AF but makes the tensor a nice cube
                    ]
                    )
            state = np.repeat([state],7,0).transpose()
            obs_list.append(torch.Tensor(np.stack((infected,dead,state))).unsqueeze(0))

        return obs_list

    def parseaction(self,a):
        key = [None,'confinement','isolation','hospital','vaccinate']
        return key[a]

    # Execute one time step within the environment
    def step(self, action):
        
        self.day += 1
        for _id, c in enumerate(self.dyn.cities):
            self.dyn.toggle(self.parseaction(action[_id]),c)
        _obs_dict = self.dyn.step()
        
        obs = self.get_obs(_obs_dict)
        self.last_obs = obs
        self.reward = []
        for c in self.dyn.cities:
            self.reward.append(self.compute_city_reward(c,_obs_dict))

        self.total_reward += np.sum(np.array(self.reward)) # sum up the individual agent rewards and compute a cumulative episode reward
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
            
        _obs_dict = self.dyn.step() # Eyo c'est un tuple ça
        self.last_obs = self.get_obs(_obs_dict)
        return self.last_obs, {'parameters':self.dyn.epidemic_parameters(self.day)}

    # Render the environment to the screen 
    def render(self, mode='human', close=False):
        total, _ = self.dyn.epidemic_parameters(self.day)
        print('Epidemic state : \n   - dead: {}\n   - infected: {}'.format(total['dead'],total['infected']))
        
        
class CountryWideEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, source_file, ep_len=100, single_control=True):
        super(CountryWideEnv, self).__init__()    
        
        self.ep_len = ep_len
        self.single_control = single_control
        self.dyn = ModelDynamics(source_file) # create the dynamical model
        
        if single_control:
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(
                            low=0, 
                            high=1, 
                            shape=(2, self.dyn.n_cities, self.dyn.env_step_length), 
                            dtype=np.float16)
        else:
            N_DISCRETE_ACTIONS = 5 #4#2**4
            self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS) # toggle behavior
            self.observation_space = spaces.Box(
                            low=0, 
                            high=1, 
                            shape=(3, self.dyn.n_cities, self.dyn.env_step_length), 
                            dtype=np.float16)


        
        self.reward = torch.Tensor([0]).unsqueeze(0)
        self.reset()
    

    def compute_reward(self, city, obs_dict): # TODO fix that so that the full country is taken into account
        dead = 0
        conf = 0
        for city in self.dyn.cities:
            dead +=  1e5 * obs_dict['city']['dead'][city][-1] 
            conf +=  150 * int(self.dyn.c_confined[city])*obs_dict['pop'][city]
        rew = (500 - dead - conf ) / (1e4 * self.dyn.total_pop)
        return torch.Tensor([rew]).unsqueeze(0)

    def get_obs(self, obs):
        infected = SCALE*np.array([np.array(obs['city']['infected'][c])/obs['pop'][c] for c in self.dyn.cities])
        dead = SCALE*np.array([np.array(obs['city']['dead'][c])/obs['pop'][c] for c in self.dyn.cities])
        return torch.Tensor(np.stack((infected,dead))).unsqueeze(0)

    def parseaction(self,a):
        if self.single_control:
            return {
                'confinement': a==1,
                'isolation': False,
                'hospital': False,
                'vaccinate': False,
            }
        else:
            raise Exception('Not implemented yet')

    # Execute one time step within the environment
    def step(self, action):
        self.day += 1
        for c in self.dyn.cities:
            self.dyn.set_action(self.parseaction(action),c)
        _obs_dict = self.dyn.step()
        self.last_obs = self.get_obs(_obs_dict)

        self.reward = self.compute_reward(c,_obs_dict)

        self.total_reward += self.reward
        done = self.day >= self.ep_len
        return self.last_obs, self.reward, done, {'parameters':self.dyn.epidemic_parameters(self.day)}

    # Reset the state of the environment to an initial state
    def reset(self, seed = None):
        self.day = 0
        self.total_reward = 0
        self.dyn.reset()
        if seed is None:
            self.dyn.start_epidemic(dt.now())
        else:
            self.dyn.start_epidemic(seed)
            
        _obs_dict = self.dyn.step() # Eyo c'est un tuple ça
        self.last_obs = self.get_obs(_obs_dict)
        return self.last_obs, {'parameters':self.dyn.epidemic_parameters(self.day)}

    # Render the environment to the screen 
    def render(self, mode='human', close=False):
        total, _ = self.dyn.epidemic_parameters(self.day)
        print('Epidemic state : \n   - dead: {}\n   - infected: {}'.format(total['dead'],total['infected']))
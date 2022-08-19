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
        
        # action space 
        N_DISCRETE_ACTIONS = 4
        self.action_space = spaces.Discrete(2**N_DISCRETE_ACTIONS)


        # the observation space is of shape N_CHANNEL x N_CITIES x N_DAYS
        #                                 = (deaths + infected) x N_CITIES x 7
        #                                 = 2 x N_CITIES x 7
        # note that values are normalized between 0 and 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, self.dyn.n_cities, self.dyn.env_step_length), dtype=np.float16)
        self.reward = torch.Tensor([0]).unsqueeze(0)
        self.reset()
    
    # Compute a reward
    def compute_reward(self, act_dict, obs_dict):
        dead_penality = 1e5 * obs_dict['total']['dead'][-1]/self.dyn.total_pop
        confinement_penality = 150*np.dot(
            np.array([int(e) for (_,e) in act_dict['confinement'].items()]), 
            np.array([float(e) for (_,e) in obs_dict['pop'].items()]))/self.dyn.total_pop
        isolation_penality = 50*np.dot(
            np.array([int(e) for (_,e) in act_dict['isolation'].items()]), 
            np.array([float(e) for (_,e) in obs_dict['pop'].items()]))/self.dyn.total_pop
        hospital_penality = 200*np.dot(
            np.array([int(e) for (_,e) in act_dict['hospital'].items()]), 
            np.array([float(e) for (_,e) in obs_dict['pop'].items()]))/self.dyn.total_pop
        vaccination_penality = 50* int(act_dict['vaccinate'])
        
        rew = (2000 - dead_penality - confinement_penality - isolation_penality - hospital_penality - vaccination_penality) / 1e5
        
        return torch.Tensor([rew]).unsqueeze(0)
    
    # converts an action to an action dictionary
    def vec2dict(self, act):
        act_digits = '{0:04b}'.format(act)
        act_dict = self.dyn.NULL_ACTION
        act_dict['confinement'] = {e:(act_digits[0] == '1') for (e,k) in act_dict['confinement'].items()}
        act_dict['isolation'] = {e:(act_digits[1] == '1') for (e,k) in act_dict['isolation'].items()}
        act_dict['hospital'] = {e:(act_digits[2] == '1') for (e,k) in act_dict['hospital'].items()}
        act_dict['vaccinate'] = (act_digits[3] == '1')
        return act_dict
    
    # converts a dictionary of observations to a normalized observation vector
    def dict2vec(self, obs):
        infected = SCALE*np.array([np.array(obs['city']['infected'][c])/obs['pop'][c] for c in self.dyn.cities])
        dead = SCALE*np.array([np.array(obs['city']['dead'][c])/obs['pop'][c] for c in self.dyn.cities])
        return torch.Tensor(np.stack((infected,dead))).unsqueeze(0)

    # Execute one time step within the environment
    def step(self, action):
        
        self.day += 1
        
        _act_dict = self.vec2dict(action)
        _obs_dict = self.dyn.step(_act_dict)
        
        obs = self.dict2vec(_obs_dict)
        self.reward = self.compute_reward(_act_dict, _obs_dict)
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
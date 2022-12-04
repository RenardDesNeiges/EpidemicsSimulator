"""Distributed version of the environement."""

import gym
import torch
from gym import spaces
import numpy as np
from epidemic_env.dynamics import ModelDynamics
from datetime import datetime as dt

from epidemic_env.env import ACTION_CONFINE, ACTION_ISOLATE, ACTION_HOSPITAL, ACTION_VACCINATE, SCALE, CONST_REWARD, DEATH_COST, ANN_COST, ISOL_COST, CONF_COST, VACC_COST, HOSP_COST

"""Custom Environment that subclasses gym env (for distributed multi agent learning)"""
class DistributedEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, source_file, ep_len=30, mode='binary'):
        super(DistributedEnv, self).__init__()

        self.ep_len = ep_len
        self.dyn = ModelDynamics(source_file)  # create the dynamical model
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
            self.action_space = spaces.Discrete(5)  # 4 actions + do nothing
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(3, self.dyn.n_cities, self.dyn.env_step_length),
                dtype=np.float16)
        else:
            raise Exception(NotImplemented)

        # this reward is not used for learning but give a comparaison metric to what happends in the non-distributed setting
        self.reward = torch.Tensor([0]).unsqueeze(0)
        self.rewards = {c: torch.Tensor([0]).unsqueeze(0)
                        for c in self.dyn.cities}
        self.reset()

    def compute_reward(self, city, obs_dict):

        def compute_death_cost():
            if city is not None:
                if len(obs_dict['city']['dead'][city]) > 1:
                    dead = DEATH_COST * (obs_dict['city']['dead'][city][-1] - obs_dict['city']
                                         ['dead'][city][-2]) / (self.dyn.map.nodes[city]['pop'])
                else:
                    dead = DEATH_COST * obs_dict['city']['dead'][city][-1] / \
                        ((self.dyn.map.nodes[city]['pop']))
                return dead
            else:
                dead = 0
                for _city in self.dyn.cities:
                    if len(obs_dict['city']['dead'][_city]) > 1:
                        dead += DEATH_COST * \
                            (obs_dict['city']['dead'][_city][-1] - obs_dict['city']
                             ['dead'][_city][-2]) / (self.dyn.total_pop)
                    else:
                        dead += DEATH_COST * \
                            obs_dict['city']['dead'][_city][-1] / \
                            (self.dyn.total_pop)
                return dead

        def compute_isolation_cost():
            if city is not None:
                isol = ISOL_COST * int(self.dyn.c_isolated[city] == self.dyn.isolation_effectiveness) * \
                    obs_dict['pop'][city] / ((self.dyn.map.nodes[city]['pop']))
                return isol
            else:
                isol = 0
                for _city in self.dyn.cities:
                    isol += ISOL_COST * \
                        int(self.dyn.c_isolated[_city] == self.dyn.isolation_effectiveness) * \
                        obs_dict['pop'][_city] / (self.dyn.total_pop)
                return isol

        def compute_confinement_cost():
            if city is not None:
                conf = CONF_COST * int(self.dyn.c_confined[city] == self.dyn.confinement_effectiveness) * \
                    obs_dict['pop'][city] / ((self.dyn.map.nodes[city]['pop']))
                return conf
            else:
                conf = 0
                for _city in self.dyn.cities:
                    conf += CONF_COST * \
                        int(self.dyn.c_confined[_city] == self.dyn.confinement_effectiveness) * \
                        obs_dict['pop'][_city] / (self.dyn.total_pop)
                return conf

        def compute_annoucement_cost():
            if city is not None:
                announcement = 0
                if self.get_info()['action']['confinement'][city] and not self.last_info['action']['confinement'][city]:
                    announcement += ANN_COST
                if self.get_info()['action']['isolation'][city] and not self.last_info['action']['isolation'][city]:
                    announcement += ANN_COST
                if self.get_info()['action']['vaccinate'][city] and not self.last_info['action']['vaccinate'][city]:
                    announcement += ANN_COST
                return announcement
            else:
                announcement = 0
                for _city in self.dyn.cities:
                    if self.get_info()['action']['confinement'][_city] and not self.last_info['action']['confinement'][_city]:
                        announcement += ANN_COST * \
                            (obs_dict['pop'][_city] / self.dyn.total_pop)
                    if self.get_info()['action']['isolation'][_city] and not self.last_info['action']['isolation'][_city]:
                        announcement += ANN_COST * \
                            (obs_dict['pop'][_city] / self.dyn.total_pop)
                    if self.get_info()['action']['vaccinate'][_city] and not self.last_info['action']['vaccinate'][_city]:
                        announcement += ANN_COST * \
                            (obs_dict['pop'][_city] / self.dyn.total_pop)
                return announcement

        def compute_vaccination_cost():
            if city is not None:
                vacc = int(self.dyn.vaccinate[city] != 0) * VACC_COST
                return vacc
            else:
                vacc = 0
                for _city in self.dyn.cities:
                    vacc += (obs_dict['pop'][_city] / self.dyn.total_pop) * \
                        int(self.dyn.vaccinate[_city] != 0) * VACC_COST
                return vacc

        def compute_hospital_cost():
            if city is not None:
                hosp = (self.dyn.extra_hospital_beds[city] != 1)*HOSP_COST
                return hosp
            else:
                hosp = 0
                for _city in self.dyn.cities:
                    hosp += (obs_dict['pop'][_city] / self.dyn.total_pop) * \
                        (self.dyn.extra_hospital_beds['Lausanne']
                         != 1) * HOSP_COST
                return hosp

        dead = compute_death_cost()
        conf = compute_confinement_cost()
        ann = compute_annoucement_cost()
        vacc = compute_vaccination_cost()
        hosp = compute_hospital_cost()
        isol = compute_isolation_cost()
        rew = CONST_REWARD - dead - conf - ann - vacc - hosp
        return torch.Tensor([rew]).unsqueeze(0), dead, conf, ann, vacc, hosp, isol

    def get_obs(self, obs):
        if self.mode == 'binary':
            infected = SCALE * \
                np.array([np.array(obs['city']['infected'][c]) /
                         obs['pop'][c] for c in self.dyn.cities])
            dead = SCALE * \
                np.array([np.array(obs['city']['dead'][c])/obs['pop'][c]
                         for c in self.dyn.cities])
            return {c: torch.Tensor(np.stack((infected, dead))).unsqueeze(0) for c in self.dyn.cities}
        elif self.mode == 'toggle':
            infected = SCALE * \
                np.array([np.array(obs['city']['infected'][c]) /
                         obs['pop'][c] for c in self.dyn.cities])
            dead = SCALE * \
                np.array([np.array(obs['city']['dead'][c])/obs['pop'][c]
                         for c in self.dyn.cities])
            confined = {c: np.ones_like(
                dead)*int((self.dyn.c_confined[c] != 1)) for c in self.dyn.cities}
            return {c: torch.Tensor(np.stack((infected, dead, confined[c]))).unsqueeze(0) for c in self.dyn.cities}
        elif self.mode == 'multi':
            infected = SCALE * \
                np.array([np.array(obs['city']['infected'][c]) /
                         obs['pop'][c] for c in self.dyn.cities])
            dead = SCALE * \
                np.array([np.array(obs['city']['dead'][c])/obs['pop'][c]
                         for c in self.dyn.cities])
            self_obs = {
                c: np.concatenate((
                    np.ones((1, 7)) * int((self.dyn.c_confined[c] != 1)),
                    np.ones((1, 7)) * int((self.dyn.c_isolated[c] != 1)),
                    np.ones((1, 7)) * int((self.dyn.vaccinate[c] != 0)),
                    np.ones((1, 7)) *
                    int((self.dyn.extra_hospital_beds[c] != 1)),
                    np.zeros((5, 7))
                )) for c in self.dyn.cities
            }
            return {c: torch.Tensor(np.stack((infected, dead, self_obs[c]))).unsqueeze(0) for c in self.dyn.cities}
        else:
            raise Exception(NotImplemented)

    def parse_action(self, a, city):
        conf = (self.dyn.c_confined[city] != 1)
        isol = (self.dyn.c_isolated[city] != 1)
        vacc = (self.dyn.vaccinate[city] != 0)
        hosp = (self.dyn.extra_hospital_beds[city] != 1)

        if self.mode == 'binary':
            conf = (a == 1)
        elif self.mode == 'toggle':
            if a == ACTION_CONFINE:
                conf = not conf
        elif self.mode == 'multi':
            if a == ACTION_CONFINE:
                conf = not conf
            elif a == ACTION_ISOLATE:
                isol = not isol
            elif a == ACTION_VACCINATE:
                vacc = not vacc
            elif a == ACTION_HOSPITAL:
                hosp = not hosp
        else:
            raise Exception(NotImplemented)

        return {
            'confinement': conf,
            'isolation': isol,
            'hospital': hosp,
            'vaccinate': vacc,
        }

    def get_info(self):
        info = {
            'parameters': self.dyn.epidemic_parameters(self.day),
            'action': {
                'confinement': {c: (self.dyn.c_confined[c] != 1)
                                for c in self.dyn.cities},
                'isolation': {c: (self.dyn.c_isolated[c] != 1)
                              for c in self.dyn.cities},
                'vaccinate': {c: (self.dyn.vaccinate[c] != 0)
                              for c in self.dyn.cities},
                'hospital': {c: (self.dyn.extra_hospital_beds[c] != 1)
                             for c in self.dyn.cities},
            },
            'dead_cost': self.dead_cost,
            'conf_cost': self.conf_cost,
            'ann_cost': self.ann_cost,
            'vacc_cost': self.vacc_cost,
            'hosp_cost': self.hosp_cost,
        }
        return info

    # Execute one time step within the environment
    def step(self, actions):

        self.day += 1
        self.last_actions = actions
        self.last_info = self.get_info()
        for c in self.dyn.cities:
            self.dyn.set_action(self.parse_action(actions[c][0], c), c)
        _obs_dict = self.dyn.step()
        self.last_obs = self.get_obs(_obs_dict)

        self.rewards = {}
        for c in self.dyn.cities:
            reward, _, _, _, _, _, _ = self.compute_reward(
                c, _obs_dict)
            self.rewards[c] = reward

        self.reward, self.dead_cost, self.conf_cost, self.ann_cost, self.vacc_cost, self.hosp_cost, self.isol_cost = self.compute_reward(
            None, _obs_dict)

        done = self.day >= self.ep_len

        return self.last_obs, float(self.reward.detach()[0]), self.rewards, done, self.get_info()

    # Reset the state of the environment to an initial state
    def reset(self, seed=None):
        self.last_action = {c: 0 for c in self.dyn.cities}
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

        _obs_dict = self.dyn.step()  # Eyo c'est un tuple ça
        self.last_obs = self.get_obs(_obs_dict)
        self.last_info = self.get_info()
        return self.last_obs, self.last_info

    # Render the environment to the screen
    def render(self, mode='human', close=False):
        total, _ = self.dyn.epidemic_parameters(self.day)
        print('Epidemic state : \n   - dead: {}\n   - infected: {}'.format(
            total['dead'], total['infected']))

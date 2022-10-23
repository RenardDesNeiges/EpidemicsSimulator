from epidemic_env.env import CountryWideEnv, DistributedEnv
from epidemic_env.visualize import Visualize
from deep_q_learning.agent import Agent
import deep_q_learning.model as models
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from datetime import datetime
import numpy as np
from copy import deepcopy

LOG_FOLDER = 'runs/'

PARAMS = {   
    'COUNTRY_WIDE_BINARY' : {
        'log' : True,
        'mode' : 'binary',
        'run_name' : 'country_wide_binary_agent' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config' : 'config/switzerland.yaml',
        'model' : 'DQN', 
        'target_update_rate' : 5,
        'reward_sample_rate' : 1,
        'viz_sample_rate' : 10,
        'num_episodes' : 300,
        'criterion' :  nn.HuberLoss(),
        'lr' :  5e-3,
        'epsilon': 0.7, 
        'epsilon_decrease': 200, 
        'gamma': 0.7,
        'buffer_size': 10000, 
        'batch_size': 512,
    },
    'COUNTRY_WIDE_BINARY_TOGGLE' : {
        'log' : True,
        'mode' : 'toggle',
        'run_name' : 'country_wide_binary_toggle_agent' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config' : 'config/switzerland.yaml',
        'model' : 'DQN', 
        'target_update_rate' : 5,
        'reward_sample_rate' : 1,
        'viz_sample_rate' : 10,
        'num_episodes' : 300,
        'criterion' :  nn.HuberLoss(),
        'lr' :  5e-3,
        'epsilon': 0.7, 
        'epsilon_decrease': 200, 
        'gamma': 0.7,
        'buffer_size': 10000, 
        'batch_size': 512
    },
    'COUNTRY_WIDE_MULTI_TOGGLE' : {
        'log' : True,
        'mode' : 'multi',
        'run_name' : 'country_wide_multiaction_agent' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config' : 'config/switzerland.yaml',
        'model' : 'DQN', 
        'target_update_rate' : 5,
        'reward_sample_rate' : 1,
        'viz_sample_rate' : 10,
        'num_episodes' : 600,
        'criterion' :  nn.HuberLoss(),
        'lr' :  5e-3,
        'epsilon': 0.7, 
        'epsilon_decrease': 550, 
        'gamma': 0.9,
        'buffer_size': 10000, 
        'batch_size': 512
    },
    'DISTRIBUTED_BINARY' : {
        'log' : True,
        'mode' : 'binary',
        'run_name' : 'decentralized_binary_agents' + datetime.today().strftime('%m_%d.%H_%M_%S'),
        'env_config' : 'config/switzerland.yaml',
        'model' : 'DQN', 
        'target_update_rate' : 5,
        'reward_sample_rate' : 1,
        'viz_sample_rate' : 10,
        'num_episodes' : 300,
        'criterion' :  nn.HuberLoss(),
        'lr' :  5e-3,
        'epsilon': 0.7, 
        'epsilon_decrease': 200, 
        'gamma': 0.7,
        'buffer_size': 10000, 
        'batch_size': 512,
    },
}


def getTrainer(name):
    if name == 'CountryWideTrainer':
        return CountryWideTrainer
    elif name == 'DistributedTrainer':
        return DistributedTrainer
    else:
        raise ValueError('Invalid trainer object!')

def getParams(name):
    if name not in PARAMS.keys():
        raise ValueError('Invalid Parameters')
    else:
        return PARAMS[name]


class DistributedTrainer():
    
    @staticmethod
    def render_log(writer,hist, episode):
        print('Logging image')
        policy_img = Visualize.render_episode_country(hist).transpose(2,0,1)
        state_img = Visualize.render_episode_fullstate(hist).transpose(2,0,1)
        cities_img = Visualize.render_episode_city(hist).transpose(2,0,1)
        writer.add_image(f'Episode/PolicyView', policy_img, episode)
        writer.add_image(f'Episode/StateView', state_img, episode)
        writer.add_image(f'Episode/CityView', cities_img, episode)
    
    @staticmethod
    def log_init(info, obs):
        info_hist = [info]
        obs_hist =  [obs]
        rew_hist = []
        loss_hist = []
        Q_hist = []
        return rew_hist, info_hist, obs_hist, loss_hist, Q_hist
    
    def log_hist(rew,info, obs, loss, _actions, rew_hist, info_hist, obs_hist, loss_hist, Q_hist):
        rew_hist.append({c:r.detach().numpy()[0,0] for c,r in rew.items()})
        obs_hist.append(obs)
        info_hist.append(info)
        loss_hist.append(loss)
        Q_hist.append({c:v[1] for c,v in _actions.items()})
    
    @staticmethod
    def tb_log(writer, episode, params,R_hist, info_hist, obs_hist, info, obs_next,loss_hist,agents,Q_hist):
        
        obs_hist.append(obs_next)
        info_hist.append(info)
        if  episode%params['reward_sample_rate'] == params['reward_sample_rate']-1:
            writer.add_scalar('Alg/Avg_Reward', 
                            np.mean([np.mean([v for _,v in r.items()]) for r in R_hist]), episode)
            writer.add_scalar('Alg/Avg_Loss', 
                            np.mean([np.mean([v for _,v in l.items()]) for l in loss_hist]), episode)
            writer.add_scalar('Alg/Avg_Q_values', 
                            np.mean([np.mean([v for _,v in q.items()]) for q in Q_hist]), episode)
            writer.add_scalar('Alg/epsilon', 
                        agents[list(agents.keys())[0]].epsilon, episode)
            writer.add_scalar('RewardShaping/dead_cost', 
                            np.mean([e['dead_cost'] for e in info_hist[:-1]]), episode)
            writer.add_scalar('RewardShaping/conf_cost', 
                            np.mean([e['conf_cost'] for e in info_hist[:-1]]), episode)
            if params['mode'] == 'toggle' or params['mode'] == 'multi':
                writer.add_scalar('RewardShaping/ann_cost', 
                            np.mean([e['ann_cost'] for e in info_hist[:-1]]), episode)
            if params['mode'] == 'multi':
                writer.add_scalar('RewardShaping/vacc_cost', 
                            np.mean([e['vacc_cost'] for e in info_hist[:-1]]), episode)
                writer.add_scalar('RewardShaping/hosp_cost', 
                            np.mean([e['hosp_cost'] for e in info_hist[:-1]]), episode)
            writer.add_scalar('RewardShaping/conf_dead_ration', 
                            np.mean([e['conf_cost'] for e in info_hist[:-1]])/np.mean([e['dead_cost'] for e in info_hist[:-1]]), episode)
            writer.add_scalar('System/Deaths', 
                            info_hist[-1]['parameters'][0]['dead'], episode)
            writer.add_scalar('System/PeakInfection', 
                            np.max([e['parameters'][0]['infected'] for e in info_hist[:-1]]), episode)
            writer.add_scalar('System/Recovered', 
                            info_hist[-1]['parameters'][0]['recovered'], episode)
            writer.add_scalar('System/ConfinedDays', 
                            np.sum([np.sum([int(c) for c in list(e['action']['confinement'].values())]) for e in info_hist[:-1]])*7, episode)
            writer.add_scalar('System/IsolationDays', 
                            np.sum([np.sum([int(c) for c in list(e['action']['isolation'].values())]) for e in info_hist[:-1]])*7, episode)
            writer.add_scalar('System/AdditionalHospitalDays', 
                            np.sum([np.sum([int(c) for c in list(e['action']['hospital'].values())]) for e in info_hist[:-1]])*7, episode)
            writer.add_scalar('System/FreeVaccinationDays', 
                            np.sum([np.sum([int(c) for c in list(e['action']['vaccinate'].values())]) for e in info_hist[:-1]])*7, episode)
    
    @staticmethod
    def train(env,agents,writer,params):
        
        for episode in range(params['num_episodes']):
                    
            if episode % params['target_update_rate'] == 0: 
                for _, agent in agents.items(): 
                    agent.targetModel.load_state_dict(agent.model.state_dict())
                
            if episode%params['viz_sample_rate'] == 0:  
                for _, agent in agents.items(): 
                    agent.epsilon = 0
            else: 
                for _, agent in agents.items(): 
                    agent.epsilon = max(params['epsilon'] - params['epsilon'] * episode/params['epsilon_decrease'],0)
            
            finished = False
            obs, info = env.reset()
            R_hist, obs_hist, info_hist, loss_hist, Q_hist = DistributedTrainer.log_init(obs,info)
            
            while not finished:
                _actions = {c:agent.act(obs) for (c,agent) in agents.items()}
                obs_next, R, finished, info = env.step(_actions)
                for c in env.dyn.cities:
                    # TODO : double check the passing of the action in memory
                    agents[c].memory.push(obs, _actions[c][0], obs_next, R[c])
                
                _losses = {c:agent.optimize_model() for (c,agent) in agents.items()}
                DistributedTrainer.log_hist(R,info, obs_next, _losses, _actions,R_hist, info_hist, obs_hist, loss_hist,Q_hist)

                obs = obs_next
                if finished:
                    break
            
            print("episode {}, avg total reward = {}, epsilon = {}".format(
                episode,np.mean([np.sum(list(rew.values())) for rew in R_hist]), 
                agents[list(agents.keys())[0]].epsilon
                ))
            if params['log']: 
                DistributedTrainer.tb_log( writer, episode, params, R_hist, info_hist, obs_hist, info, obs_next,loss_hist,agents, Q_hist)
                if episode%params['viz_sample_rate'] == 0:  
                    DistributedTrainer.render_log(writer,info_hist, episode)


        return None    
    
    @staticmethod
    def run(params):
        
        if not params['log']:
            print('WARNING LOGGING IS NOT ENABLED, NO TB LOGS OF THE EXPERIMENT WILL BE SAVED')
        
        logpath = LOG_FOLDER+params['run_name']
        
        env = DistributedEnv(params['env_config'], mode=params['mode'])
        if hasattr(models, params['model']):
            model = getattr(models, params['model'])
        else:
            print(f'Error : {params["model"]} is not a valid model name,')
            return None
        
        _agent = Agent(  env = env, 
                        model = model, 
                        criterion = params['criterion'], 
                        lr = params['lr'], 
                        epsilon = params['epsilon'], 
                        gamma = params['gamma'],
                        buffer_size = params['buffer_size'],
                        batch_size = params['batch_size']
                    )
        agents = {c:deepcopy(_agent) for c in env.dyn.cities}
    
        if params['log']:
            writer = SummaryWriter(log_dir=logpath)
        else:
            writer = None

        DistributedTrainer.train(env,agents,writer,params)
        

class CountryWideTrainer():
    
    @staticmethod
    def render_log(writer,hist, episode):
        print('Logging image')
        policy_img = Visualize.render_episode_country(hist).transpose(2,0,1)
        state_img = Visualize.render_episode_fullstate(hist).transpose(2,0,1)
        cities_img = Visualize.render_episode_city(hist).transpose(2,0,1)
        writer.add_image(f'Episode/PolicyView', policy_img, episode)
        writer.add_image(f'Episode/StateView', state_img, episode)
        writer.add_image(f'Episode/CityView', cities_img, episode)
    
    @staticmethod
    def log_init(info, obs):
        info_hist = [info]
        obs_hist =  [obs]
        rew_hist = []
        loss_hist = []
        Q_hist = []
        return rew_hist, info_hist, obs_hist, loss_hist, Q_hist
    
    def log_hist(rew,info, obs, loss, distrib, rew_hist, info_hist, obs_hist, loss_hist, Q_hist):
        rew_hist.append(rew.detach().numpy()[0,0])
        obs_hist.append(obs)
        info_hist.append(info)
        loss_hist.append(loss)
        Q_hist.append(distrib)
    
    @staticmethod
    def tb_log(writer, episode, params,R_hist, info_hist, obs_hist, info, obs_next,loss_hist,agent,Q_hist):
        
        obs_hist.append(obs_next)
        info_hist.append(info)
        if  episode%params['reward_sample_rate'] == params['reward_sample_rate']-1:
            writer.add_scalar('Alg/Avg_Reward', 
                            np.mean(R_hist), episode)
            writer.add_scalar('Alg/Loss', 
                            np.mean(loss_hist), episode)
            writer.add_scalar('Alg/Avg_Q_values', 
                            np.mean(Q_hist), episode)
            writer.add_scalar('Alg/epsilon', 
                        agent.epsilon, episode)
            writer.add_scalar('RewardShaping/dead_cost', 
                            np.mean([e['dead_cost'] for e in info_hist[:-1]]), episode)
            writer.add_scalar('RewardShaping/conf_cost', 
                            np.mean([e['conf_cost'] for e in info_hist[:-1]]), episode)
            if params['mode'] == 'toggle' or params['mode'] == 'multi':
                writer.add_scalar('RewardShaping/ann_cost', 
                            np.mean([e['ann_cost'] for e in info_hist[:-1]]), episode)
            if params['mode'] == 'multi':
                writer.add_scalar('RewardShaping/vacc_cost', 
                            np.mean([e['vacc_cost'] for e in info_hist[:-1]]), episode)
                writer.add_scalar('RewardShaping/hosp_cost', 
                            np.mean([e['hosp_cost'] for e in info_hist[:-1]]), episode)
            writer.add_scalar('RewardShaping/conf_dead_ration', 
                            np.mean([e['conf_cost'] for e in info_hist[:-1]])/np.mean([e['dead_cost'] for e in info_hist[:-1]]), episode)
            writer.add_scalar('System/Deaths', 
                            info_hist[-1]['parameters'][0]['dead'], episode)
            writer.add_scalar('System/PeakInfection', 
                            np.max([e['parameters'][0]['infected'] for e in info_hist[:-1]]), episode)
            writer.add_scalar('System/Recovered', 
                            info_hist[-1]['parameters'][0]['recovered'], episode)
            writer.add_scalar('System/ConfinedDays', 
                            np.sum([np.sum(e['action']['confinement']) for e in info_hist[:-1]])*7, episode)
            writer.add_scalar('System/IsolationDays', 
                            np.sum([np.sum(e['action']['isolation']) for e in info_hist[:-1]])*7, episode)
            writer.add_scalar('System/AdditionalHospitalDays', 
                            np.sum([np.sum(e['action']['hospital']) for e in info_hist[:-1]])*7, episode)
            writer.add_scalar('System/FreeVaccinationDays', 
                            np.sum([np.sum(e['action']['vaccinate']) for e in info_hist[:-1]])*7, episode)
    
    @staticmethod
    def train(env,agent,writer,params):
        
        for episode in range(params['num_episodes']):
                    
            if episode % params['target_update_rate'] == 0: 
                agent.targetModel.load_state_dict(agent.model.state_dict())

            if episode%params['viz_sample_rate'] == 0:  
                agent.epsilon = 0
            else: 
                agent.epsilon = max(params['epsilon'] - params['epsilon'] * episode/params['epsilon_decrease'],0)
            
            finished = False
            obs, info = env.reset()
            R_hist, obs_hist, info_hist, loss_hist, Q_hist = CountryWideTrainer.log_init(obs,info)
            
            while not finished:
                action, est_Q = agent.act(obs)
                obs_next, R, finished, info = env.step(action)
                agent.memory.push(obs, action, obs_next, R)
                
                loss = agent.optimize_model()
                CountryWideTrainer.log_hist(R,info, obs_next, loss, est_Q,R_hist, info_hist, obs_hist, loss_hist,Q_hist)

                obs = obs_next
                if finished:
                    break
            
            print("episode {}, avg reward = {}, epsilon = {}".format(episode,np.mean(R_hist), agent.epsilon))
            if params['log']: 
                CountryWideTrainer.tb_log( writer, episode, params, R_hist, info_hist, obs_hist, info, obs_next,loss_hist,agent, Q_hist)
                if episode%params['viz_sample_rate'] == 0:  
                    CountryWideTrainer.render_log(writer,info_hist, episode)


        return None    
    
    @staticmethod
    def run(params):
        
        if not params['log']:
            print('WARNING LOGGING IS NOT ENABLED, NO TB LOGS OF THE EXPERIMENT WILL BE SAVED')
        
        logpath = LOG_FOLDER+params['run_name']
        
        env = CountryWideEnv(params['env_config'], mode=params['mode'])
        if hasattr(models, params['model']):
            model = getattr(models, params['model'])
        else:
            print(f'Error : {params["model"]} is not a valid model name,')
            return None
        
        agent = Agent(  env = env, 
                        model = model, 
                        criterion = params['criterion'], 
                        lr = params['lr'], 
                        epsilon = params['epsilon'], 
                        gamma = params['gamma'],
                        buffer_size = params['buffer_size'],
                        batch_size = params['batch_size']
                    ) #
        
        if params['log']:
            writer = SummaryWriter(log_dir=logpath)
        else:
            writer = None

        CountryWideTrainer.train(env,agent,writer,params)
        
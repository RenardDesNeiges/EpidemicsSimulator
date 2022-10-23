from epidemic_env.env import EpidemicEnv, CountryWideEnv
from epidemic_env.visualize import Visualize
from deep_q_learning.agent import Agent
import deep_q_learning.model as models
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from datetime import datetime
import numpy as np
from copy import deepcopy

LOG_FOLDER = 'runs/'

DEFAULT_PARAMS = {
    'log' : True,
    'run_name' : 'default' + datetime.today().strftime('%m_%d.%H_%M_%S'),
    'env_config' : 'config/switzerland.yaml',
    'model' : 'DQN', 
    'target_update_rate' : 5,
    'reward_sample_rate' : 1,
    'viz_sample_rate' : 10,
    'num_episodes' : int(3e4),
    'criterion' :  nn.MSELoss(),
    'lr' :  5e-3,
    'epsilon': 0.7, 
    'epsilon_decrease': 200, 
    'gamma': 0.7,
    'buffer_size': 10000, 
    'batch_size': 512
}

class Trainer():
    
    @staticmethod
    def render_log(writer,hist, episode):
        print('Logging image')
        episode_img = Visualize.render_episode_country(hist).transpose(2,0,1)
        writer.add_image(f'Episode/Viz', episode_img, episode)
        pass
    
    @staticmethod
    def train(env,agent,writer,params):

        cumulative_reward = 0
        cumulative_loss = 0
        
        # create one agent per city : 
        agents = {}
        for c in env.dyn.cities:
            agents[c] = deepcopy(agent)

        for episode in range(params['num_episodes']):
            finished = False
            S, info = env.reset()
            info_hist = [info]
            obs_hist = [S]
            
            while not finished:
                a = []
                for _id,c in enumerate(env.dyn.cities):
                    obs = S[_id]
                    a.append(agents[c].act(obs))
                info_hist[-1]['action'] = {
                    'confinement': [env.dyn.c_confined[c] for c in env.dyn.cities],
                    'isolation': [env.dyn.c_isolated[c] for c in env.dyn.cities],
                    'hospital': [env.dyn.extra_hospital_beds[c] for c in env.dyn.cities],
                    'vaccinate': [env.dyn.vaccinate[c] for c in env.dyn.cities],
                }
                
                Sp, R, finished, info = deepcopy(env.step(a)) # TODO : compute a single action from every agent's action
                # TODO : split the rewards up
                loss = 0
                obs_hist.append(Sp)
                info_hist.append(info)
                for _id,_ in enumerate(env.dyn.cities):
                    agents[c].memory.push(S[_id], a[_id], Sp[_id], R[_id])
                    if episode % params['target_update_rate'] == 0:
                        agents[c].targetModel.load_state_dict(agents[c].model.state_dict())
                    if episode%params['viz_sample_rate'] == 0:
                        agents[c].epsilon = 0
                    else:
                        agents[c].epsilon = params['epsilon']
                    loss += agents[c].optimize_model()

                cumulative_reward = env.total_reward
                S = Sp
                cumulative_loss += loss
                if finished:
                    break
                    
            if  episode%params['reward_sample_rate'] == params['reward_sample_rate']-1:
                
                if params['log']: 
                    writer.add_scalar('Train/Reward', 
                                    float((cumulative_reward/params['reward_sample_rate'])[0,0]), episode)
                    writer.add_scalar('Train/Loss', 
                                    cumulative_loss/params['reward_sample_rate'], episode)
                    writer.add_scalar('Model/Deaths', 
                                    info_hist[-1]['parameters'][0]['dead'], episode)
                    writer.add_scalar('Model/PeakInfection', 
                                    np.max([e['parameters'][0]['infected'] for e in info_hist[:-1]]), episode)
                    writer.add_scalar('Model/Recovered', 
                                    info_hist[-1]['parameters'][0]['recovered'], episode)
                    writer.add_scalar('Model/ConfinedDays', 
                                    np.sum([np.sum(e['action']['confinement']) for e in info_hist[:-1]])*7, episode)
                    writer.add_scalar('Model/IsolationDays', 
                                    np.sum([np.sum(e['action']['isolation']) for e in info_hist[:-1]])*7, episode)
                    writer.add_scalar('Model/AdditionalHospitalDays', 
                                    np.sum([np.sum(e['action']['hospital']) for e in info_hist[:-1]])*7, episode)
                    writer.add_scalar('Model/FreeVaccinationDays', 
                                    np.sum([np.sum(e['action']['vaccinate']) for e in info_hist[:-1]])*7, episode)

                print("episode {}, avg reward = {}".format(episode, float((cumulative_reward/params['reward_sample_rate'])[0,0])))

                cumulative_reward = 0
                cumulative_loss = 0        

                # if episode%params['viz_sample_rate'] == 0:
                #     if params['log']:
                #         Trainer.render_log(writer, info_hist,episode)

        return None    
    
    @staticmethod
    def run (params):
        
        logpath = LOG_FOLDER+params['run_name']
        
        env = EpidemicEnv(params['env_config'])
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
                    )
        
        if params['log']:
            writer = SummaryWriter(log_dir=logpath)
        
        Trainer.train(env,agent,writer,params)
        

class CountryWideTrainer():
    
    @staticmethod
    def render_log(writer,hist, episode):
        print('Logging image')
        episode_img = Visualize.render_episode_country(hist).transpose(2,0,1)
        writer.add_image(f'Episode/Viz', episode_img, episode)
        pass
    
    @staticmethod
    def log_init(info, obs):
        info_hist = [info]
        obs_hist =  [obs]
        rew_hist = []
        loss_hist = []
        distrib_hist = []
        return rew_hist, info_hist, obs_hist, loss_hist, distrib_hist
    
    def log_hist(rew,info, obs, loss, distrib, rew_hist, info_hist, obs_hist, loss_hist, distrib_hist):
        rew_hist.append(rew.detach().numpy()[0,0])
        obs_hist.append(obs)
        info_hist.append(info)
        loss_hist.append(loss)
        distrib_hist.append(distrib)
    
    @staticmethod
    def tb_log(writer, episode, params,R_hist, info_hist, obs_hist, info, obs_next,cumulative_reward,loss_hist,agent,distrib_hist):
        
        obs_hist.append(obs_next)
        info_hist.append(info)
        if  episode%params['reward_sample_rate'] == params['reward_sample_rate']-1:
            writer.add_scalar('Alg/Avg_Reward', 
                            np.mean(R_hist), episode)
            writer.add_scalar('Alg/Loss', 
                            np.mean(loss_hist), episode)
            writer.add_scalar('Alg/mean_distrib', 
                            np.mean(distrib_hist), episode)
            writer.add_scalar('Alg/epsilon', 
                        agent.epsilon, episode)
            writer.add_scalar('RewardShaping/dead_cost', 
                            np.mean([e['dead_cost'] for e in info_hist[:-1]]), episode)
            writer.add_scalar('RewardShaping/conf_cost', 
                            np.mean([e['conf_cost'] for e in info_hist[:-1]]), episode)
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

        print("episode {}, avg reward = {}, epsilon = {}".format(episode,np.mean(R_hist), agent.epsilon))
    
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
            R_hist, obs_hist, info_hist, loss_hist, distrib_hist = CountryWideTrainer.log_init(obs,info)
            
            while not finished:
                action, distrib = agent.act(obs)
                obs_next, R, finished, info = env.step(action)
                agent.memory.push(obs, action, obs_next, R)
                
                loss = agent.optimize_model()
                CountryWideTrainer.log_hist(R,info, obs_next, loss, distrib,R_hist, info_hist, obs_hist, loss_hist,distrib_hist)

                obs = obs_next
                if finished:
                    break
                            
            if params['log']: 
                CountryWideTrainer.tb_log( writer, episode, params, R_hist, info_hist, obs_hist, info, obs_next,env.total_reward,loss_hist,agent, distrib_hist)
                if episode%params['viz_sample_rate'] == 0:  
                    CountryWideTrainer.render_log(writer,info_hist, episode)


        return None    
    
    @staticmethod
    def run(params):
        
        logpath = LOG_FOLDER+params['run_name']
        
        env = CountryWideEnv(params['env_config'])
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
                    )
        
        if params['log']:
            writer = SummaryWriter(log_dir=logpath)
        
        CountryWideTrainer.train(env,agent,writer,params)
        
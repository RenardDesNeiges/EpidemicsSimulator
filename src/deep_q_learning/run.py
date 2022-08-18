from epidemic_env.epidemic_env import EpidemicEnv
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
    'run_name' : 'default' + datetime.today().strftime('%m_%d.%H_%M_%S'),
    'env_config' : 'config/switzerland.yaml',
    'model' : 'DQN',
    'target_update_rate' : 5,
    'reward_sample_rate' : 1,
    'viz_sample_rate' : 10,
    'num_episodes' : 1000,
    'criterion' :  nn.MultiLabelSoftMarginLoss(),
    'lr' :  5e-4,
    'epsilon': 0.3, 
    'gamma': 0.99,
    'buffer_size': 10000, 
    'batch_size': 64
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

        for episode in range(params['num_episodes']):
            finished = False
            S, _, _, info = env.reset()
            info_hist = [info]
            obs_hist = [S]
            a_buffer = []
            
            while not finished:
                a = agent.act(S) 
                info_hist[-1]['action'] = deepcopy(env.vec2dict(a))
            
                Sp, R, finished, info = deepcopy(env.step(a))
                obs_hist.append(Sp)
                info_hist.append(info)
                a_buffer.append(np.array([int(e) for e in '{0:04b}'.format(a) ]))
                agent.memory.push(S, a, Sp, R)
                
                cumulative_reward += R
                S = Sp
                
                if episode % params['target_update_rate'] == 0:
                    agent.targetModel.load_state_dict(agent.model.state_dict())

                loss = agent.optimize_model()
                cumulative_loss += loss
                if finished:
                    break
                
            if  episode%params['reward_sample_rate'] == params['reward_sample_rate']-1:

                writer.add_scalar('Train/Reward', 
                                  float((cumulative_reward/params['reward_sample_rate'])[0,0]), episode)
                writer.add_scalar('Train/Loss', 
                                  cumulative_loss/params['reward_sample_rate'], episode)

                writer.add_scalar('Model/Deaths', 
                                  cumulative_loss/info_hist[-1]['parameters'][0]['dead'], episode)

                writer.add_scalar('Model/Recovered', 
                                  cumulative_loss/info_hist[-1]['parameters'][0]['recovered'], episode)

                print("episode {}, avg reward = {}".format(episode, float((cumulative_reward/params['reward_sample_rate'])[0,0])))

                cumulative_reward = 0
                cumulative_loss = 0        

            if episode%params['viz_sample_rate'] == 0:
                Trainer.render_log(writer, info_hist,episode)
                pass

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
                        criterion =params['criterion'], 
                        lr = params['lr'], 
                        epsilon = params['epsilon'], 
                        gamma = params['gamma'],
                        buffer_size = params['buffer_size'],
                        batch_size = params['batch_size'])
        
        writer = SummaryWriter(log_dir=logpath)
        
        Trainer.train(env,agent,writer,params)
        
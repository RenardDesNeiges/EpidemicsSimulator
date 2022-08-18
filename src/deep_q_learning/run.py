from epidemic_env.epidemic_env import EpidemicEnv
from deep_q_learning.agent import Agent
import deep_q_learning.model as models
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from datetime import datetime

LOG_FOLDER = 'runs/log/'

DEFAULT_PARAMS = {
    'run_name' : 'default' + datetime.today().strftime('%m_%d.%H_%M_%S'),
    'env_config' : 'config/switzerland.yaml',
    'model' : 'DQN',
    'target_update_rate' : 5,
    'reward_sample_rate' : 2,
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
    def train(env,agent,writer,params):

        cumulative_reward = 0
        cumulative_loss = 0

        for episode in range(params['num_episodes']):
            finished = False
            S, _, _, info = env.reset()
            info_hist = [info]
            
            while not finished:
                a = agent.act(S) 
            
                Sp, R, finished, info = env.step(a) 
                info_hist.append(info)
                
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

                print("episode {}, avg reward = {}".format(episode, float((cumulative_reward/params['reward_sample_rate'])[0,0])))

                cumulative_reward = 0
                cumulative_loss = 0        

            if  episode%params['viz_sample_rate'] == params['viz_sample_rate']:

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
        
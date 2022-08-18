from epidemic_env.epidemic_env import EpidemicEnv
from agent import Agent
from model import DQN

DEFAULT_CONFIG = 'config/switzerland.yaml'

def run (env_path = DEFAULT_CONFIG):
    print('run')
    
    env = EpidemicEnv(env_path)
    model = DQN()
    params = None
    
    agent = Agent(env, model, params)
    pass
    
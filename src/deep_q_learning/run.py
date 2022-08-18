from epidemic_env.epidemic_env import EpidemicEnv
from train import Trainer

DEFAULT_CONFIG = 'config/switzerland.yaml'

def run (env_path = DEFAULT_CONFIG):
    print('run')
    
    env = EpidemicEnv(env_path)
    model = None
    params = None
    
    train = Trainer(env, model, params)
    pass
    
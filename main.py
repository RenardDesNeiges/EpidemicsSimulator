import sys
sys.path.append('./src')
from src.epidemic_env.epidemic_env import EpidemicEnv
from src.deep_q_learning.run import run

DEFAULT_CONFIG = 'config/switzerland.yaml'

if __name__ == '__main__':

    run('config/switzerland.yaml')
    
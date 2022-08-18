import sys
sys.path.append('./src')

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from src.epidemic_env.epidemic_env import EpidemicEnv
from src.deep_q_learning.run import Trainer

DEFAULT_CONFIG = 'config/switzerland.yaml'

if __name__ == '__main__':

    Trainer.run('config/switzerland.yaml')
    
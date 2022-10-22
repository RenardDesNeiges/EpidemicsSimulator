#  /Users/renard/miniconda3/bin/python

import sys
sys.path.append('./src')

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from src.epidemic_env.env import EpidemicEnv
from src.deep_q_learning.run import Trainer, CountryWideTrainer, DEFAULT_PARAMS

DEFAULT_CONFIG = 'config/switzerland.yaml'

if __name__ == '__main__':
    # Trainer.run(DEFAULT_PARAMS)
    CountryWideTrainer.run(DEFAULT_PARAMS)
    
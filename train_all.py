#  /Users/renard/miniconda3/bin/python


import os
import sys
sys.path.append('./src')

from src.deep_q_learning.run import getTrainer, getParams

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


DEFAULT_CONFIG = 'config/switzerland.yaml'


def run(trainer_name, params_name):
    _trainer = getTrainer(trainer_name)
    _params = getParams(params_name)
    print(
        f"Training the agent with trainer {trainer_name} and parameters {params_name}")
    _trainer.run(_params)


if __name__ == '__main__':
   run('CountryWideTrainer', 'COUNTRY_WIDE_BINARY')
   run('CountryWideTrainer', 'COUNTRY_WIDE_BINARY_TOGGLE')
   run('CountryWideTrainer', 'COUNTRY_WIDE_MULTI_TOGGLE')
   run('DistributedTrainer', 'DISTRIBUTED_BINARY')
   run('DistributedTrainer', 'DISTRIBUTED_BINARY_TOGGLE')
   run('DistributedTrainer', 'DISTRIBUTED_MULTI_TOGGLE')

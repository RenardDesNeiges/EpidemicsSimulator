#  /Users/renard/miniconda3/bin/python
import os

from deep_q_learning import getTrainer, getParams

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


DEFAULT_CONFIG = 'config/switzerland.yaml'


def run(params_name):
    _trainer = getTrainer()
    _params = getParams(params_name)
    print(
        f"Training the agent with parameters {params_name}")
    _trainer.run(_params)


if __name__ == '__main__':
   run('COUNTRY_WIDE_BINARY')
   run('COUNTRY_WIDE_BINARY_TOGGLE')
   run('COUNTRY_WIDE_MULTI_TOGGLE')
   run('COUNTRY_WIDE_MULTI_FACTORIZED')


#  /Users/renard/miniconda3/bin/python

import sys
sys.path.append('./src')

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from src.deep_q_learning.run import CountryWideTrainer, COUNTRY_WIDE_BINARY, COUNTRY_WIDE_BINARY_TOGGLE, COUNTRY_WIDE_MULTI_TOGGLE

DEFAULT_CONFIG = 'config/switzerland.yaml'

if __name__ == '__main__':
    CountryWideTrainer.run(COUNTRY_WIDE_MULTI_TOGGLE)
    
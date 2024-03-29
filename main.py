#  /Users/renard/miniconda3/bin/python

import sys
sys.path.append('./src')

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import argparse

from deep_q_learning import getTrainer, getParams

DEFAULT_CONFIG = 'config/switzerland.yaml'

if __name__ == '__main__':
    parser = argparse.ArgumentParser() # create the parser object

    parser.add_argument("--trainer", type=str,  default='CountryWideTrainer',
                        help="Give the trainer object to be used")
    parser.add_argument("--params", type=str,  default='COUNTRY_WIDE_MULTI_TOGGLE',
                        help="Give the trainer object to be used")

    args = parser.parse_args() # get the named tuple
    
    _trainer = getTrainer(args.trainer)
    _params = getParams(args.params)
    print(f"Training the agent with trainer {args.trainer} and parameters {args.params}")
    _trainer.run(_params)